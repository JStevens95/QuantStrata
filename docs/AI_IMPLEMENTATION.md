# RADE Analytics — AI Assistant Implementation Guide

| Field | Value |
| --- | --- |
| **Status** | Design — implementation pending |
| **Scope** | `rade_analytics` Dash UI + `ensemble.api` FastAPI backend |
| **LLM provider** | OpenAI (assumed; switchable via single client class) |
| **Intended readers** | Quants, platform engineers, product stakeholders |
| **Last updated** | 2026-06-09 |

---

## Table of Contents

- [Part 1 — Generic AI + Dash Integration](#part-1--generic-ai--dash-integration)
  - [1.1 Why integrate AI into a Dash UI](#11-why-integrate-ai-into-a-dash-ui)
  - [1.2 Architectural pattern](#12-architectural-pattern)
  - [1.3 The five core components](#13-the-five-core-components)
  - [1.4 Request lifecycle in theory](#14-request-lifecycle-in-theory)
  - [1.5 The six capability tiers](#15-the-six-capability-tiers)
  - [1.6 Per-tier requirements](#16-per-tier-requirements)
  - [1.7 Compliance and audit](#17-compliance-and-audit)
  - [1.8 Cost and rate-limit considerations](#18-cost-and-rate-limit-considerations)
  - [1.9 Knowledge augmentation (RAG over technical docs)](#19-knowledge-augmentation-rag-over-technical-docs)
- [Part 2 — `rade_analytics` Integration](#part-2--rade_analytics-integration)
  - [2.1 Where the AI assistant lives in the architecture](#21-where-the-ai-assistant-lives-in-the-architecture)
  - [2.2 Backend script structure (FastAPI)](#22-backend-script-structure-fastapi)
  - [2.3 Frontend script structure (Dash)](#23-frontend-script-structure-dash)
  - [2.4 Configuration changes](#24-configuration-changes)
  - [2.5 Component interaction diagram](#25-component-interaction-diagram)
  - [2.6 Request lifecycle — concrete example](#26-request-lifecycle--concrete-example)
  - [2.7 Page Contract compliance](#27-page-contract-compliance)
  - [2.8 Activity log integration](#28-activity-log-integration)
  - [2.9 Document index for `rade_analytics`](#29-document-index-for-rade_analytics)
- [Part 3 — Capabilities Catalogue](#part-3--capabilities-catalogue)
  - [3.1 Stakeholder personas](#31-stakeholder-personas)
  - [3.2 Tier 1 — Conversational data access](#32-tier-1--conversational-data-access)
  - [3.3 Tier 2 — Cross-source synthesis](#33-tier-2--cross-source-synthesis)
  - [3.4 Tier 3 — Reasoning and explanation](#34-tier-3--reasoning-and-explanation)
  - [3.5 Tier 4 — Generative actions](#35-tier-4--generative-actions)
  - [3.6 Tier 5 — Proactive and agentic](#36-tier-5--proactive-and-agentic)
  - [3.7 Tier 6 — Domain-differentiating capabilities](#37-tier-6--domain-differentiating-capabilities)
  - [3.8 Prioritisation matrix](#38-prioritisation-matrix)
- [Part 4 — Mock Visualisations](#part-4--mock-visualisations)
- [Appendix A — OpenAI API specifics](#appendix-a--openai-api-specifics)
- [Appendix B — Tool registry skeleton](#appendix-b--tool-registry-skeleton)
- [Appendix C — System prompt templates](#appendix-c--system-prompt-templates)
- [Appendix D — Implementation phase plan](#appendix-d--implementation-phase-plan)
- [Appendix E — RAG ingestion pipeline reference](#appendix-e--rag-ingestion-pipeline-reference)

---

# Part 1 — Generic AI + Dash Integration

## 1.1 Why integrate AI into a Dash UI

A modern analytics dashboard like `rade_analytics` exposes dozens of API endpoints, hundreds of charts, and tens of thousands of cells of data. Even an experienced user is limited by:

- **Discoverability** — they can only ask what they remember the dashboard supports.
- **Click cost** — answering "what's the worst cluster this week, and why?" requires opening 3–4 tabs, sorting tables, eyeballing charts, and synthesising mentally.
- **Cross-context reasoning** — joining drift data with inference results with governance history is a manual, error-prone task done in Excel.
- **Onboarding** — new joiners take weeks to learn where each chart lives.

An AI assistant inverts the model: the user describes the *outcome* they want in natural language, and the system orchestrates the existing endpoints to deliver it. It is not a replacement for the dashboard — it is a **second navigation layer** that complements clicking with describing.

The implementation pattern is well-established (Anthropic's Claude in Slack, OpenAI's ChatGPT plugins, Linear's AI assistant, Notion AI, Hex's Magic). The architectural shape is consistent: a side panel in the UI talking to a backend orchestrator that uses tool-calling to compose the existing API.

## 1.2 Architectural pattern

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌────────────────────────┐                                                  │
│  │ Dash UI (browser)      │                                                  │
│  │                        │      POST /ai/chat (streaming)                   │
│  │  ┌──────────────────┐  │      ─────────────────────────►                  │
│  │  │ AI side panel    │  │                                                  │
│  │  │  - chat thread   │  │      ◄─────────────────────────                  │
│  │  │  - input box     │  │      SSE stream: text chunks, tool_use,         │
│  │  │  - inline chart  │  │                   tool_result, complete         │
│  │  │  - context chips │  │                                                  │
│  │  └──────────────────┘  │                                                  │
│  │                        │                                                  │
│  └────────────────────────┘                                                  │
│                                                                              │
│         ▲                                                                    │
│         │ (renders)                                                          │
│         │                                                                    │
│  ┌──────┴─────────────────────────────────────────────────────────────────┐ │
│  │ FastAPI backend                                                         │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │ │
│  │  │ routers/ai.py   │ ─► │ ai_session.py    │ ─► │ ai_client.py     │  │ │
│  │  │ (HTTP layer)    │    │ (orchestration)  │    │ (LLM HTTP I/O)   │  │ │
│  │  └─────────────────┘    └──────────────────┘    └──────────────────┘  │ │
│  │                                  │                                      │ │
│  │                                  ▼                                      │ │
│  │                          ┌──────────────────┐                           │ │
│  │                          │ ai_tools.py      │  registry + dispatcher    │ │
│  │                          └──────────────────┘                           │ │
│  │                                  │                                      │ │
│  │                  ┌───────────────┼───────────────┐                      │ │
│  │                  ▼               ▼               ▼                      │ │
│  │           ┌───────────┐   ┌───────────┐   ┌───────────┐                 │ │
│  │           │ /clusters │   │ /portfolio│   │ /monitoring│   …            │ │
│  │           │  router   │   │  router   │   │   router   │                │ │
│  │           └───────────┘   └───────────┘   └───────────┘                 │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐    ┌──────────────────┐                            │ │
│  │  │ ai_audit.py     │    │ ai_conversations │  sqlite or in-memory       │ │
│  │  │ (compliance)    │    │ (state mgmt)     │                            │ │
│  │  └─────────────────┘    └──────────────────┘                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌──────────────────┐                                │
│                          │ OpenAI API       │  api.openai.com or             │
│                          │ (or work proxy)  │  enterprise gateway            │
│                          └──────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

Key architectural principles:

1. **The LLM never touches the database directly.** It calls *your* API endpoints as tools. This means every AI answer goes through the same auth, audit, and validation layers as a regular UI click — no new attack surface or data exfiltration path.
2. **Tools are dumb proxies.** Each tool maps 1:1 to an existing endpoint. No new business logic. This keeps the AI integration testable in isolation and ensures parity between "what the AI can see" and "what the UI shows".
3. **Streaming is non-optional.** A 4-second wait for a synchronous LLM response feels broken; the same response streamed feels instant. Build with SSE from day one.
4. **Conversation state lives on the backend.** The browser only holds a `conversation_id`. This keeps prompts auditable and lets the user reopen old chats.
5. **Context is injected automatically.** Every request silently includes "user is on the X tab, looking at cluster Y, in ensemble version Z" so the user never has to repeat themselves.

## 1.3 The five core components

| Component | Responsibility | Lives in |
| --- | --- | --- |
| **LLM client** | Thin HTTP wrapper around the OpenAI API (or work proxy). Handles auth, retries, timeouts, streaming primitive. | `services/ai_client.py` |
| **Tool registry** | List of tool definitions (name, description, JSON schema for input). Each tool has a dispatch handler that calls an existing API endpoint or service. | `services/ai_tools.py` |
| **Session orchestrator** | Multi-turn loop: send message + tools to LLM → receive tool_use → dispatch → feed tool_result back → repeat until text-only response. Streams events out via async generator. | `services/ai_session.py` |
| **Router** | HTTP layer: defines the endpoints (`/ai/chat`, `/ai/chat/stream`, `/ai/investigate/{id}`, etc.). Validates input, calls session orchestrator, emits SSE. | `routers/ai.py` |
| **UI panel** | Dash component: floating pill + collapsible drawer + chat thread + input box + inline chart renderer. Streams responses via `EventSource` (clientside callback) or polling. | `layouts/ai_panel.py` + `callbacks/ai_assistant_cb.py` |

These five components are the **complete** scope of an AI assistant integration. Everything else (specialised endpoints, proactive triggers, reports) is built *on top* of these five.

## 1.4 Request lifecycle in theory

A single user message flows through the system like this:

```
┌──────────┐   1. user types message                ┌────────────────┐
│ Browser  │ ─────────────────────────────────────► │ ai_panel cb    │
└──────────┘                                        └────────────────┘
                                                            │
                                                            │ 2. POST /ai/chat/stream
                                                            │    {conversation_id, text, page_context}
                                                            ▼
                                                    ┌────────────────┐
                                                    │ routers/ai.py  │
                                                    └────────────────┘
                                                            │
                                                            │ 3. handoff to async generator
                                                            ▼
                                                    ┌────────────────┐
                                                    │ ai_session     │
                                                    │   load history │
                                                    │   inject ctx   │
                                                    └────────────────┘
                                                            │
                                                            │ 4. call OpenAI with tools
                                                            ▼
                                                    ┌────────────────┐
                                                    │ OpenAI API     │
                                                    └────────────────┘
                                                            │
                          ┌─────────────────────────────────┤
                          │ 5a. text chunks                 │
                          │     (stream straight through)   │
                          │                                 │
                          │ 5b. tool_use chunk              │
                          ▼                                 │
                  ┌───────────────┐                         │
                  │ ai_tools      │                         │
                  │   dispatch    │ ─► calls /clusters/cl-7/summary
                  └───────────────┘                         │
                          │                                 │
                          │ 6. tool_result fed back         │
                          │                                 │
                          ▼                                 │
                  ┌───────────────┐                         │
                  │ OpenAI API    │ ──── repeat from 5 ────►│
                  │ (continues)   │                         │
                  └───────────────┘                         │
                                                            │
                                                            │ 7. SSE chunks
                                                            ▼
                                                    ┌────────────────┐
                                                    │ Browser (Dash) │
                                                    │   incremental  │
                                                    │   render       │
                                                    └────────────────┘
                                                            │
                                                            │ 8. persist final message
                                                            ▼
                                                    ┌────────────────┐
                                                    │ ai_audit       │
                                                    │ conversations  │
                                                    └────────────────┘
```

The critical loop is between steps 5 and 6: the LLM can call multiple tools across multiple turns, all transparent to the user. From the browser's perspective, one question produces one streaming response — even if the backend made 7 tool calls to assemble it.

## 1.5 The six capability tiers

| Tier | Theme | One-line description | Risk | WOW factor |
| --- | --- | --- | --- | --- |
| **Tier 1** | Conversational data access | "Tell me X" — retrieve and display existing data | Low | Low–Medium |
| **Tier 2** | Cross-source synthesis | "Join data from multiple tabs to answer a question" | Low | Medium |
| **Tier 3** | Reasoning and explanation | "Why is X happening?" — produce hypotheses with evidence | Medium | High |
| **Tier 4** | Generative actions | "Build / draft / propose X" — produces artefacts the user reviews | Medium | High |
| **Tier 5** | Proactive and agentic | AI acts without being asked — briefings, alerts, investigations | High | Very high |
| **Tier 6** | Domain-differentiating | Capabilities specific to your codebase that off-the-shelf AI cannot do | High | Very high |

The tiers build on each other architecturally:
- T1 needs only the basic five components.
- T2 needs T1 + a richer tool registry covering all major endpoints.
- T3 needs T2 + structured output schemas + careful system prompts.
- T4 needs T3 + write-capable tools + approval flows.
- T5 needs T4 + background workers + push notifications.
- T6 needs T5 + deep integration with model internals and domain artefacts.

## 1.6 Per-tier requirements

| Tier | Backend | Frontend | Compliance | Dev effort |
| --- | --- | --- | --- | --- |
| **Tier 1** | LLM client, single `/ai/chat` endpoint, 6–10 read tools | Side panel, chat thread, simple text rendering | Audit log of prompts + responses | 1–2 weeks |
| **Tier 2** | Expanded tool registry (~20 tools), multi-turn orchestration | Inline tables, citation chips | (same as T1) | +1 week |
| **Tier 3** | Structured output schemas (Pydantic), reasoning system prompts, evidence-citing requirement | Inline charts (Plotly JSON), evidence cards, structured rendering | Citation auditing — every claim links to a tool call | +2 weeks |
| **Tier 4** | Write-capable tools (gated), approval flow endpoints, draft persistence | Review modal, edit-before-confirm UX, draft library | Human-in-the-loop gating, mandatory review log | +2 weeks |
| **Tier 5** | Background job runner (e.g. `apscheduler`), webhook receivers, push channel | Notification badge, briefing card on landing, opt-in scheduler UI | Per-user trigger preferences, "do not page" hours, all auto-actions logged | +3 weeks |
| **Tier 6** | Hooks into model internals (GNN attention, RNN states), governance graph queries, lineage queries | Specialised rendering for each capability (explainability heatmaps, lineage trees) | (domain-specific) | +4–6 weeks |

## 1.7 Compliance and audit

Even though OpenAI's enterprise tier offers strong data handling guarantees, a quant-finance shop will typically require an additional audit and compliance layer:

| Requirement | Implementation |
| --- | --- |
| **Prompt logging** | Every prompt + response written to `ai_audit_log` table (sqlite or PostgreSQL) with timestamp, user, model, token counts, and a hash of the prompt for later replay |
| **PII / position redaction** | Pre-LLM redactor that masks trade IDs, counterparty names, etc. before sending. Post-response un-redactor restores them. Implemented as middleware around `ai_client.complete()` |
| **Read-only by default** | The tool registry has a `mutates: bool` field on each tool. Mutating tools require an explicit "approval_token" in the request, granted by a separate UI confirmation step |
| **Rate limiting** | Per-user, per-day token cap (e.g. 100k tokens/user/day) and per-org cost cap. Enforced in `ai_session` before LLM call |
| **Right to be forgotten** | `DELETE /ai/conversations/{id}` purges both the conversation store and the audit log entries (within legal retention requirements) |
| **Hallucination indicator** | Every answer ends with `confidence: high | medium | low` based on whether all claims were backed by tool results. Low-confidence answers display a warning banner |

## 1.8 Cost and rate-limit considerations

OpenAI pricing (as of 2026) makes "always-on" AI assistant viable but not free. Plan for:

- **Average chat turn cost** — 1k–5k input tokens + 200–1k output tokens. At `gpt-4o` prices, ~$0.01–$0.05 per turn.
- **Tool-heavy turns** — 5–10x more input tokens because tool results are appended to history. Plan for $0.05–$0.50 per "investigate this cluster" turn.
- **Daily user budget** — 50 chat turns per active user = ~$2.50/day. A team of 20 active users = $50/day = $15k/year.
- **Background briefing cost** — 1 morning briefing per day per user = ~$0.20 each. Linear in users.

Mitigations:
- Use `gpt-4o-mini` for Tier 1 lookups (10x cheaper) and reserve `gpt-4o` / `o1` family for Tier 3+ reasoning.
- Cache common tool results (e.g. "latest ensemble version") for 5 minutes.
- Trim conversation history aggressively — keep only the last 10 turns by default, with explicit "extend context" option.
- For background jobs, batch into hourly windows and use the OpenAI Batch API for 50% discount.

## 1.9 Knowledge augmentation (RAG over technical docs)

Tools (Section 1.3) give the AI access to *live data*. **Retrieval-Augmented Generation (RAG)** gives it access to *documented knowledge* — methodology, framework guides, pipeline contracts, model risk write-ups — so it can answer questions like "how does the residual MLP work?" or "what's our PSI threshold methodology?" with citations to the actual project documentation.

### 1.9.1 Why RAG, not fine-tuning

Two common approaches to give an LLM domain knowledge:

| Aspect | Fine-tuning | RAG |
| --- | --- | --- |
| **Cost** | Re-train per significant doc change ($$$) | Cents per query, near-zero index update |
| **Update latency** | Hours/days | Seconds (re-embed the changed doc) |
| **Citation / traceability** | Impossible — knowledge is baked into weights | Mandatory — every claim links to its source chunk |
| **Hallucination control** | Worse — model improvises confidently | Better — model is constrained to "answer only from these chunks" |
| **Versioning** | Painful — which doc version is in the weights? | Metadata-driven — `doc.version=v2025-12-01` |
| **Compliance** | "How did the model conclude that?" → unanswerable | "Show me the paragraph it cited" → trivial |
| **Best for** | Style, tone, vocabulary — *not* facts | Facts, definitions, methodology lookups |

For regulated quant finance use cases where every claim should be auditable, **RAG wins on every dimension**. The only argument for fine-tuning is teaching the AI house writing style for MRM documents — and even that is usually better solved with a few-shot example in the system prompt.

### 1.9.2 The RAG pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Build-time (one-off + incremental)                                       │
│                                                                          │
│   docs/**/*.md  ─────►  chunker  ─────►  embedder  ─────►  vector store  │
│   src/**/README                          (OpenAI or local)               │
│   *.tex paper                                                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│ Query-time (every relevant chat turn)                                    │
│                                                                          │
│   user question  ─►  LLM decides: call search_docs(query, top_k)         │
│                                          │                               │
│                                          ▼                               │
│                                  embed(query)  ─►  vector store          │
│                                          │                               │
│                                          ▼                               │
│                                  top-k chunks  ─►  (optional) re-rank    │
│                                          │                               │
│                                          ▼                               │
│                                  return chunks + citations               │
│                                          │                               │
│                                          ▼                               │
│                                  LLM answers, citing chunks              │
└──────────────────────────────────────────────────────────────────────────┘
```

The query-time path is **one additional tool call** in the existing session loop — no architectural changes to `ai_session.py` or the streaming machinery.

### 1.9.3 Document tiering

Not all docs are equal — methodology pages should be retrievable for serious investigation questions, while operational READMEs are more useful for "how do I use the dashboard?" questions. Tag every chunk with a tier so the retriever can filter:

| Tier | Examples | Used for |
| --- | --- | --- |
| **A — Model / methodology** | `reference/models/*.md`, `paper/section_data_pipeline.tex`, `inference_pipeline_contract.md`, `RADE_ML_IMPLEMENTATION_PLAN.md` | "How does the hybrid GNN-RNN work?", "What's our PSI methodology?", "Why do we standardise residuals?" |
| **B — Architecture / pipelines** | `RADE_UI_DESIGN.md`, `PIPELINE.md`, `page_contract.md`, `ensemble_implementation.md` | "How is inference structured?", "What's the page contract?", "Where do scenario shocks get injected?" |
| **C — Framework guides** | `guides/machine_learning/*`, `guides/risk/risk_framework.md`, `guides/instruments/*` | "How do we price X?", "What's our risk framework?", onboarding |
| **D — User / operational** | `guides/ui/dash_ui_tutorial.md`, `BEST_PRACTICES.md`, READMEs | "How do I use the dashboard?", "What are our coding conventions?" |

The tier filter is a tool input — the LLM (steered by the system prompt) chooses which tiers to search per query. Methodology questions hit A; "how do I…" questions hit C and D.

### 1.9.4 Chunking strategy

Chunking is the single highest-leverage decision. Three options:

| Strategy | How it works | Pro | Con |
| --- | --- | --- | --- |
| **Fixed-size** | 500-token chunks with 50-token overlap | Trivial to implement | Splits mid-section; loses semantic boundaries |
| **Markdown-aware** | Split on `##` headers; preserve section context in each chunk | Section semantics preserved; small implementation cost | Long sections need sub-chunking |
| **Semantic** | LLM-assisted boundary detection | Best recall | Costs $0.01–$0.05 per doc to chunk |

**Recommended starter: markdown-aware**. The `LangChain` `MarkdownHeaderTextSplitter` does this in 10 lines. Each chunk carries the H1/H2/H3 chain as metadata so the retriever sees `Hybrid GNN-RNN › Residual MLP › Standardisation` and the LLM can quote the section.

### 1.9.5 Vector store options

Three tiers based on scale:

| Option | Setup | Best for | Caveats |
| --- | --- | --- | --- |
| **In-memory `numpy`** | 50 LoC, no dependencies | <10k chunks (basically your whole repo today) | Lost on restart; not multi-process |
| **`chromadb` local file** | One pip install, transparent persistence | 10k–1M chunks, single-host | Adds a file-system dep, but zero ops |
| **`pgvector` on existing DB** | SQL extension | When you want SQL joins with other tables | Needs PostgreSQL |
| **Hosted (`pinecone`, `weaviate`)** | API account | Multi-region, large team | Cost + data egress |

For `rade_analytics` today: **chromadb** is the sweet spot — persistent, zero ops, and the index sits next to the rest of the API state.

### 1.9.6 Embedding model options

| Model | Cost (per 1M tokens) | Recall | Notes |
| --- | --- | --- | --- |
| OpenAI `text-embedding-3-small` (1536d) | $0.02 | Good | Pragmatic default |
| OpenAI `text-embedding-3-large` (3072d) | $0.13 | Best | 6x cost, ~10% recall gain |
| Local `sentence-transformers/all-MiniLM-L6-v2` (384d) | Free | Decent | No data leaves; recall ~20% below OpenAI |
| Local `BAAI/bge-large-en-v1.5` (1024d) | Free | Good | Bigger local model; on par with `3-small` |

For compliance-sensitive docs, the local model is the right answer even at modest recall cost — only the retrieved chunks leave your network with the LLM call, not the entire corpus during indexing.

### 1.9.7 Hallucination control

The cheapest hallucination defence is in the system prompt:

```
When you call search_docs, your answer MUST be derived from the returned chunks.
Quote the source: include `[<doc>, §<section>]` after each factual claim.
If the chunks do not answer the question, say so plainly — do not improvise.
```

The expensive but worthwhile defence is **citation validation**: after the LLM finishes, parse its citations and verify each `[doc, section]` reference actually appears in the retrieved chunks. Discard or flag any uncited claims.

### 1.9.8 Compliance considerations specific to RAG

RAG sends document chunks to OpenAI in every relevant prompt. For sensitive docs the same policy questions as for prompts apply, at higher volume:

| Mitigation | What it does |
| --- | --- |
| **`confidentiality` metadata** | Tag docs (`public` / `internal` / `restricted`); retriever filters to allowed levels per user |
| **Local embedding** | Embedding lookup is on-prem; only retrieved chunks leave |
| **Local LLM fallback** | Sensitive-tier queries route to an on-prem model; OpenAI skipped entirely |
| **PII redactor in the indexer** | Trade IDs, counterparty names redacted at index time, not query time |
| **Doc versioning** | When MRM doc v2 supersedes v1, index marks v1 `superseded=true` and retriever filters it out |

---

# Part 2 — `rade_analytics` Integration

## 2.1 Where the AI assistant lives in the architecture

The existing system has:

- **FastAPI backend** at `src/rade_ml_pt/ensemble/api/` exposing 17 routers (overview, clusters, portfolio, monitoring, inference, etc.).
- **Dash UI** at `src/ui/apps/rade_analytics/` with 7 tabs (Overview, Evaluation, Risk Management, Inference, Monitoring, Governance, etc.).
- **Page Contract** rules defining how callbacks must be structured (mount-tripwire pattern, captured-vs-render separation).
- **Activity log** pattern via `infer_events.py` for stage events.

The AI assistant slots in as:

- **One new router** (`ai.py`) alongside the existing 17.
- **Three new services** (`ai_client.py`, `ai_session.py`, `ai_tools.py`) alongside the existing ones.
- **One new layout** (`ai_panel.py`) mounted globally in the app shell — not a tab.
- **One new callback file** (`ai_assistant_cb.py`) following Page Contract rules.
- **One new API client method** (`chat_with_ai`) on the rade_analytics API client.

No existing files are modified except for `app.py` (mount the panel + include the router) and `api/dependencies.py` (provide the AI client). The AI integration is fully additive.

## 2.2 Backend script structure (FastAPI)

```
src/rade_ml_pt/ensemble/api/
├── app.py                            ← MODIFY: include ai_router, add startup hook
├── config.py                         ← MODIFY: add OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL
├── dependencies.py                   ← MODIFY: add get_ai_client(), get_ai_session_mgr()
├── models/
│   ├── ai.py                         ← NEW: Pydantic models for AI requests/responses
│   │   - ChatRequest, ChatResponse
│   │   - ChatStreamChunk (text, tool_use, tool_result, done, error)
│   │   - InvestigateRequest, InvestigationReport
│   │   - GenerateReportRequest, ReportDraft
│   │   - Citation, EvidenceChartSpec
│   └── ... (existing)
├── routers/
│   ├── ai.py                         ← NEW: AI HTTP endpoints
│   │   POST /prism/v1/ai/chat            — synchronous single-turn (testing)
│   │   POST /prism/v1/ai/chat/stream     — SSE streaming, multi-turn
│   │   POST /prism/v1/ai/investigate/{monitoring_run_id}
│   │   POST /prism/v1/ai/summarise/inference/{inference_run_id}
│   │   POST /prism/v1/ai/generate-report
│   │   GET  /prism/v1/ai/conversations/{conv_id}
│   │   GET  /prism/v1/ai/conversations            — list user's conversations
│   │   DELETE /prism/v1/ai/conversations/{conv_id}
│   │   GET  /prism/v1/ai/health
│   │   GET  /prism/v1/ai/tools                    — list available tools (for debugging)
│   └── ... (existing 17 routers)
└── services/
    ├── ai_client.py                  ← NEW: OpenAI HTTP wrapper
    │   - Class AiClient(openai.AsyncOpenAI)
    │   - chat(messages, tools, stream) → AsyncIterator[chunk]
    │   - structured_output(messages, schema) → BaseModel
    │   - estimate_cost(messages) → float
    ├── ai_tools.py                   ← NEW: tool registry + dispatcher
    │   - TOOLS: list[ToolDef]
    │   - dispatch(name, input_dict, ctx) → ToolResult
    │   - Each tool wraps an existing service call (NOT an HTTP call)
    ├── ai_session.py                 ← NEW: multi-turn orchestrator
    │   - Class AiSessionManager (in-memory store of conversations)
    │   - run_chat(conv_id, user_msg, page_ctx) → AsyncIterator[StreamChunk]
    │   - Handles the tool-call loop until LLM stops calling tools
    │   - Persists every turn to ai_audit + conversation store
    ├── ai_audit.py                   ← NEW: compliance log writer
    │   - log_prompt(...) → audit_id
    │   - log_response(audit_id, ...)
    │   - log_tool_call(audit_id, tool_name, input, output)
    │   - Writes to sqlite table `ai_audit_log`
    ├── ai_redaction.py               ← NEW: PII/position redaction
    │   - redact(text) → (redacted_text, restore_map)
    │   - restore(text, restore_map) → original_text
    │   - Trade IDs, counterparty IDs, large notional values
    └── ... (existing services: reader.py, paths.py, cache.py, etc.)
```

### Database additions

```
src/rade_ml_pt/ensemble/db_schema.sql   ← MODIFY: add ai_audit_log + ai_conversations
```

```sql
CREATE TABLE ai_audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    user_id         TEXT,
    model           TEXT NOT NULL,
    role            TEXT NOT NULL,            -- 'user', 'assistant', 'tool'
    content_hash    TEXT NOT NULL,            -- sha256 of prompt/response
    content_preview TEXT,                     -- first 200 chars, redacted
    token_count_in  INTEGER,
    token_count_out INTEGER,
    cost_usd        REAL,
    tool_calls      TEXT,                     -- json: [{name, input_hash}, ...]
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    INDEX idx_ai_audit_conv (conversation_id),
    INDEX idx_ai_audit_user_ts (user_id, ts)
);

CREATE TABLE ai_conversations (
    id              TEXT PRIMARY KEY,         -- uuid4
    user_id         TEXT,
    title           TEXT,                     -- auto-generated from first message
    page_context    TEXT,                     -- json: starting page/cluster/run
    messages_json   TEXT NOT NULL,            -- full message history
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    archived        INTEGER NOT NULL DEFAULT 0,
    INDEX idx_ai_conv_user (user_id, updated_at DESC)
);
```

## 2.3 Frontend script structure (Dash)

```
src/ui/apps/rade_analytics/
├── app.py                            ← MODIFY: mount ai_panel in app shell
├── api_client.py                     ← MODIFY: add AI client methods
│   - chat_with_ai_stream(conv_id, message, ctx) → AsyncIterator
│   - get_ai_conversations() → list
│   - investigate_run(run_id) → InvestigationReport
├── config.py                         ← MODIFY: add AI_PANEL_DEFAULT_OPEN, AI_QUICK_PROMPTS
├── layouts/
│   ├── ai_panel.py                   ← NEW: floating pill + drawer + chat
│   │   - ai_panel_layout() → Component
│   │     - dmc.Affix wrapping a pill button (bottom-right)
│   │     - dmc.Drawer (slides from right) containing:
│   │       - Header: title, conversation chip, "New chat", "History", close
│   │       - Context chips row: [ensemble vX] [cluster cY] [tab Z]
│   │       - Chat thread: list of message bubbles
│   │       - Suggested actions row (changes per page)
│   │       - Input box + Send + slash-command hints
│   │       - Footer: model name, tool list, audit indicator
│   │   - _message_bubble(msg) → Component (renders text + inline charts/tables)
│   │   - _suggested_actions(page_context) → list[Component]
│   └── ... (existing layouts)
├── callbacks/
│   ├── ai_assistant_cb.py            ← NEW: chat callbacks
│   │   - register_callbacks(app):
│   │     - toggle_ai_drawer (pill click → open drawer)
│   │     - capture_user_message (send button / enter → server message)
│   │     - render_streaming_response (clientside_callback via EventSource)
│   │     - load_conversation (history click → reload thread)
│   │     - new_conversation (button → reset thread + new conv_id)
│   │     - update_context_chips (url change → update displayed context)
│   └── ... (existing callback files)
├── components/                       ← NEW: AI-specific small components
│   ├── ai_message_bubble.py          ← user vs assistant bubble rendering
│   ├── ai_inline_chart.py            ← Plotly chart embedded in chat
│   ├── ai_evidence_card.py           ← evidence + citation rendering for T3
│   ├── ai_investigation_card.py      ← structured investigation rendering
│   └── ai_context_chip.py            ← dismissable context indicator
├── figures/
│   ├── ai_evidence_charts.py         ← NEW: helpers that turn tool results into Plotly figs
│   └── ... (existing figures)
└── assets/
    ├── ai_panel.css                  ← NEW: styling for panel + bubbles + streaming cursor
    └── ai_panel.js                   ← NEW: EventSource handling for streaming
```

## 2.4 Configuration changes

### `src/rade_ml_pt/ensemble/api/config.py`

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # AI integration
    openai_api_key:      str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url:     str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model_chat:   str = Field(default="gpt-4o-mini", env="OPENAI_MODEL_CHAT")
    openai_model_reason: str = Field(default="gpt-4o",       env="OPENAI_MODEL_REASON")
    openai_org_id:       Optional[str] = Field(default=None, env="OPENAI_ORG_ID")

    # Audit / compliance
    ai_audit_enabled:    bool = Field(default=True, env="AI_AUDIT_ENABLED")
    ai_redaction_enabled: bool = Field(default=True, env="AI_REDACTION_ENABLED")
    ai_max_tokens_per_user_per_day: int = Field(default=100_000, env="AI_USER_TOKEN_CAP")
    ai_max_cost_per_org_per_day_usd: float = Field(default=500.0, env="AI_ORG_COST_CAP")

    # Performance
    ai_default_max_tool_iterations: int = Field(default=8, env="AI_MAX_TOOL_ITERATIONS")
    ai_default_timeout_seconds:     int = Field(default=60, env="AI_TIMEOUT_SECONDS")
```

### `src/ui/apps/rade_analytics/config.py`

```python
# AI panel
AI_PANEL_ENABLED       = True
AI_PANEL_DEFAULT_OPEN  = False
AI_PANEL_WIDTH         = 480              # px when drawer is open
AI_API_BASE            = "/prism/v1/ai"

# Per-page suggested prompts (rendered as chips above the input)
AI_QUICK_PROMPTS_BY_TAB = {
    "overview":         ["Summarise today's portfolio health",
                         "What's changed since yesterday?",
                         "Which clusters need attention?"],
    "evaluation":       ["Worst-performing 5 clusters this split",
                         "Compare to last evaluation run",
                         "Explain RMSE drivers"],
    "risk-management":  ["What's the tail loss attribution?",
                         "Show me cluster-level VaR contributions",
                         "Find scenarios with extreme losses"],
    "inference":        ["Diff this run vs the previous",
                         "Which clusters were re-priced?",
                         "Explain the predicted PnL change"],
    "monitoring":       ["Investigate this drift run",
                         "Which clusters drifted most?",
                         "Is this severity critical?"],
    "governance":       ["Show me the deployment history",
                         "Why was vX promoted?",
                         "Draft a model risk review entry"],
}
```

## 2.5 Component interaction diagram

```mermaid
graph TB
  subgraph Browser
    Pill[Floating AI pill]
    Drawer[AI side-panel drawer]
    Thread[Chat thread + inline charts]
    Input[Input box + Send]
  end

  subgraph "Dash server"
    Toggle[toggle_ai_drawer cb]
    Capture[capture_user_message cb]
    Render[render_streaming_response<br/>clientside cb / EventSource]
    APIClient[rade_analytics<br/>api_client.py]
  end

  subgraph "FastAPI app"
    Router[routers/ai.py]
    Session[ai_session.py<br/>SessionManager]
    Tools[ai_tools.py<br/>Dispatcher]
    Client[ai_client.py<br/>OpenAI client]
    Audit[ai_audit.py]
  end

  subgraph "Existing rade_ml_pt services"
    OverviewSvc[overview reader]
    ClusterSvc[cluster reader]
    MonitorSvc[monitoring reader]
    InferenceSvc[inference reader]
  end

  subgraph External
    OpenAI[OpenAI / work proxy]
    AuditDB[(sqlite ai_audit_log)]
    ConvDB[(sqlite ai_conversations)]
  end

  Pill --> Toggle --> Drawer
  Input --> Capture --> APIClient
  APIClient -->|POST /ai/chat/stream| Router
  Router --> Session
  Session --> Audit
  Audit --> AuditDB
  Session --> Client
  Client --> OpenAI
  OpenAI -.tool_use.-> Client
  Client --> Session
  Session --> Tools
  Tools --> OverviewSvc
  Tools --> ClusterSvc
  Tools --> MonitorSvc
  Tools --> InferenceSvc
  Tools --> Session
  Session -.SSE chunks.-> Router
  Router -.SSE.-> Render
  Render --> Thread
  Session --> ConvDB
```

## 2.6 Request lifecycle — concrete example

Consider the user typing `"What's the worst-performing cluster this week?"` while on the Evaluation tab, with ensemble version `v2026-05-30` loaded.

```
Step 1 — Browser
  User types message → presses Send
  Dash `Input` triggers capture_user_message callback

Step 2 — Dash callback (capture_user_message)
  Reads:
    - conversation_id from dcc.Store("ai-conversation-id")
    - page context from dcc.Store("session-store") + url.pathname
  Calls:
    api_client.chat_with_ai_stream(
      conv_id="conv-abc123",
      message="What's the worst-performing cluster this week?",
      page_ctx={"tab": "evaluation",
                "ensemble_version": "v2026-05-30",
                "selected_cluster": None,
                "selected_run": None,
                "iso_date": "2026-06-09"}
    )
  Returns: nothing (clientside callback takes over for streaming)

Step 3 — Clientside callback opens EventSource
  new EventSource("/prism/v1/ai/chat/stream?conv_id=conv-abc123&...")
  Appends an empty assistant message bubble to the thread
  Each SSE event appends a token chunk to that bubble

Step 4 — FastAPI router/ai.py
  POST /ai/chat/stream
    - Validate request (Pydantic)
    - Get session manager from dependency
    - Return StreamingResponse(ai_session.run_chat(...), media_type="text/event-stream")

Step 5 — ai_session.py: run_chat() async generator
  - audit_id = ai_audit.log_prompt(conv_id, user_msg)
  - history = load_conversation(conv_id)
  - system_prompt = build_system_prompt(page_ctx)
  - messages = [system_prompt, *history, user_msg]
  - yield {"event": "start", "audit_id": audit_id}

  Loop:
    response_iter = ai_client.chat(messages, tools=TOOLS, stream=True)
    async for chunk in response_iter:
      if chunk.type == "text":
        yield {"event": "text", "delta": chunk.text}
      elif chunk.type == "tool_use":
        tool_result = ai_tools.dispatch(chunk.name, chunk.input, ctx)
        ai_audit.log_tool_call(audit_id, chunk.name, chunk.input, tool_result)
        yield {"event": "tool_use", "name": chunk.name, "input": chunk.input}
        yield {"event": "tool_result", "name": chunk.name, "summary": tool_result.summary}
        messages.append({"role": "assistant", "tool_use": chunk})
        messages.append({"role": "tool", "tool_result": tool_result})
        # continue outer loop to call LLM again
      elif chunk.type == "done":
        ai_audit.log_response(audit_id, full_response)
        save_conversation(conv_id, messages)
        yield {"event": "done"}
        return

Step 6 — Inside the loop, an example tool dispatch:
  ai_tools.dispatch("list_clusters_with_metrics", {"sort_by": "mae_desc", "limit": 5}, ctx)
    → calls cluster_reader.list_with_metrics(ensemble_version=ctx.ensemble_version,
                                             sort_by="mae_desc", limit=5)
    → returns: [{"cluster_id": "cl-3", "mae": 0.24, ...}, ...]

Step 7 — LLM responds with text now that it has the data:
  "The worst-performing cluster this week is **cl-3** with MAE 0.24, up from 0.18 last week..."
  (streamed token by token via SSE)

Step 8 — When the LLM decides to render a chart:
  tool_use: render_chart({type: "bar", x: ["cl-3", "cl-5", "cl-7"], y: [0.24, 0.21, 0.18], ...})
  The dispatcher returns immediately; the panel renders the chart inline.

Step 9 — Browser receives "event: done" → finalises bubble, removes typing indicator
```

The user perceived 1 question, ~3 seconds, 1 streamed answer with an inline chart. The backend did 3 tool calls and 2 LLM round-trips.

## 2.7 Page Contract compliance

Following the project's established callback rules (see `docs/rade_analytics/page_contract.md`):

| Rule | How `ai_assistant_cb.py` complies |
| --- | --- |
| **L4 mount-tripwire** | Drawer's `is_open` state is captured-vs-rendered through a `dcc.Store` so reopening the panel restores the last conversation |
| **C4 pathname is State, never Input** | The chat callback reads `url.pathname` as `State` to compute `page_context`; pathname changes do NOT auto-fire chat |
| **C5 capture / render separation** | `capture_user_message` only writes the user message into a `dcc.Store("ai-pending-message")`; `render_streaming_response` is a separate clientside cb that reads the store and opens the EventSource |
| **C7 prevent_initial_call** | All AI callbacks use `prevent_initial_call=True` except the toggle (which needs to handle browser refresh state) |

## 2.8 Activity log integration

The existing `infer_events.py` activity-log pattern is reused. When the AI assistant takes an action that materially affects state (creating a scenario, drafting a report, promoting a model), the same event vocabulary is used so the action appears in the same activity log the user already trusts:

```python
# Inside an AI tool that writes
event(
    stage="ai_assistant",
    phase="Scenario draft created",
    status="ok",
    target=scenario_id,
    detail=f"prompt: {prompt[:100]}... · user={user_id}",
)
```

This means stakeholders never wonder "did the AI do that or did the user do that?" — both routes flow through the same observability stack.

## 2.9 Document index for `rade_analytics`

This section maps Section 1.9 (general RAG theory) onto the actual `rade_analytics` repository: which folders to index, where the indexer lives, how the `search_docs` tool plugs into the existing tool registry, and the re-indexing schedule.

### 2.9.1 What gets indexed

The current repository contains ~132 markdown files, 10+ READMEs, a LaTeX paper, and `PIPELINE.md` documents. The proposed source coverage:

```
Indexed in tier A (model methodology):
  docs/reference/models/**/*.md
  docs/reference/machine_learning/*.md
  docs/reference/risk/*.md
  docs/rade_ml/RADE_ML_IMPLEMENTATION_PLAN.md
  docs/rade_ml_pt_implementation_guide.md
  docs/inference_pipeline_contract.md
  docs/paper/section_data_pipeline.tex
  docs/paper/section_data_pipeline_companion.md

Indexed in tier B (architecture / pipelines):
  docs/platform_designs/RADE_UI_DESIGN.md
  docs/platform_designs/ensemble_analytics_db_v3_blueprint.md
  docs/rade_analytics/page_contract.md
  docs/rade_analytics/inference_implementation.md
  docs/rade_analytics/ensemble_infer_refactor.md
  docs/ensemble_implementation.md
  docs/ensemble_dashboard_design.md
  docs/hybrid_gnn_rnn_pipeline_suggested_changes.md
  docs/AI_IMPLEMENTATION.md            (this very document)
  src/rade_sr/PIPELINE.md
  src/rade_ml_pt/**/README.md
  src/rade_sr/README.md

Indexed in tier C (framework guides):
  docs/guides/**/*.md
  docs/architecture/**/*.md

Indexed in tier D (user / operational):
  docs/guides/ui/dash_ui_tutorial.md
  docs/BEST_PRACTICES.md
  src/ui/**/README.md
  README.md
  UPDATE.md
```

**Explicitly excluded** (curated):

```
docs/testing/**/*                  ← test failure logs, noisy
docs/development/roadmap.md        ← changes weekly; cite via tool not RAG
docs/project_assessment.md         ← internal status; not useful to AI
docs/paper/*.aux                   ← LaTeX build artefacts
docs/paper/*.out
```

### 2.9.2 Where the indexer code lives

Additive only — no existing files modified. Two new service modules, one new model file, one new CLI script:

```
src/rade_ml_pt/ensemble/api/
├── services/
│   ├── ai_doc_index.py         ← NEW: ingestion + chunking + embedding
│   │   - DocChunk dataclass (text, doc_path, section_chain, tier, version_hash)
│   │   - DocIndexer class
│   │     - ingest_directory(root, tier, exclude_globs) → n_chunks
│   │     - ingest_file(path, tier) → list[DocChunk]
│   │     - _chunk_markdown(text) → list[(section_chain, text)]
│   │     - _embed_batch(texts) → np.ndarray
│   │     - _upsert(chunks) → None
│   │     - delete_by_path(path) → None
│   │   - watch_for_changes(roots, interval_seconds=300) → background task
│   ├── ai_doc_retriever.py     ← NEW: query-side
│   │   - DocRetriever class
│   │     - search(query, top_k=5, tier=None, confidentiality=None) → list[DocChunk]
│   │     - rerank(chunks, query, top_k) → list[DocChunk]   (optional)
│   │     - format_for_llm(chunks) → str  (with [<doc>, §<section>] citations)
│   └── ai_tools.py             ← MODIFY: register search_docs tool (see Appendix B)
├── models/
│   └── ai.py                   ← MODIFY: add DocChunkModel, SearchDocsResponse
├── routers/
│   └── ai.py                   ← MODIFY: add GET /ai/docs/health
│                                         GET /ai/docs/stats
│                                         POST /ai/docs/reindex (admin only)
├── dependencies.py             ← MODIFY: add get_doc_indexer(), get_doc_retriever()
└── app.py                      ← MODIFY: kick off watch_for_changes in lifespan

scripts/
└── reindex_docs.py             ← NEW: CLI for bulk rebuild
                                   python scripts/reindex_docs.py --full
                                   python scripts/reindex_docs.py --tier A
                                   python scripts/reindex_docs.py --path docs/reference/
```

### 2.9.3 Storage layout

Chroma persists to disk. The path is configurable but the default sits next to the existing API state:

```
<project_root>/.cache/rade_ai/
├── chroma/                       ← chromadb persistent dir
│   ├── <collection_uuid>/
│   │   ├── chroma.sqlite3
│   │   └── ...
├── ingestion_log.jsonl           ← every indexed file: path, hash, ts, tier
└── index_meta.json               ← n_chunks, n_docs, last_full_rebuild, model_id
```

Add this to `.gitignore`.

### 2.9.4 Re-indexing strategy

Three triggers, in order of importance:

1. **On-demand CLI** — `python scripts/reindex_docs.py --full` for the initial build and major restructures. Runtime: ~2 minutes for the current corpus.
2. **Background file-watcher** — `apscheduler` job every 5 minutes that compares file mtimes against `ingestion_log.jsonl` and re-embeds only the changed files. <200ms overhead when nothing changed.
3. **Pre-commit hook (optional)** — re-indexes the changed docs in the developer's local copy so writers see their changes available to the assistant immediately.

Each re-index produces a row in `ingestion_log.jsonl`:

```json
{"path": "docs/reference/models/hybrid_gnn_rnn.md",
 "tier": "model_methodology",
 "n_chunks": 18,
 "hash_before": "9f3a...",
 "hash_after":  "ab21...",
 "ts": "2026-06-09T22:47:12Z",
 "embedded_in_seconds": 1.4,
 "tokens_embedded": 4811,
 "cost_usd": 0.00010}
```

### 2.9.5 Config additions

Extend the `Settings` from §2.4:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # RAG / document index
    ai_rag_enabled:        bool = Field(default=True, env="AI_RAG_ENABLED")
    ai_rag_store_path:     str  = Field(default=".cache/rade_ai/chroma", env="AI_RAG_STORE_PATH")
    ai_rag_embed_model:    str  = Field(default="text-embedding-3-small", env="AI_RAG_EMBED_MODEL")
    ai_rag_chunk_size:     int  = Field(default=800,  env="AI_RAG_CHUNK_SIZE")
    ai_rag_chunk_overlap:  int  = Field(default=120,  env="AI_RAG_CHUNK_OVERLAP")
    ai_rag_default_top_k:  int  = Field(default=5,    env="AI_RAG_DEFAULT_TOP_K")
    ai_rag_watch_interval_seconds: int = Field(default=300, env="AI_RAG_WATCH_INTERVAL")
    ai_rag_local_embeddings: bool = Field(default=False, env="AI_RAG_LOCAL_EMBEDDINGS")
    ai_rag_doc_roots:      List[str] = Field(
        default=["docs/", "src/rade_sr/PIPELINE.md", "src/**/README.md"],
        env="AI_RAG_DOC_ROOTS",
    )
```

### 2.9.6 How retrieval is woven into the chat loop

The `search_docs` tool is registered alongside the existing 12 (see Appendix B). The system prompt is extended with one paragraph to teach the LLM when to use it:

```
KNOWLEDGE TOOLS
You have access to `search_docs` for questions about how the system works,
not what it currently shows. Use it for:
  - "How does X work?" / "What is X?"
  - "Why is X structured this way?"
  - "What's our methodology for X?"
  - "Where is X defined / documented?"
For numerical or state questions ("what is cl-7's MAE?") use the data
tools instead. Cite every methodology claim as [<doc>, §<section>].
```

For investigations (Tier 3 capabilities), the prompt is stronger:

```
For each hypothesis, prefer to cite our own documentation when the
methodology is the explanation. e.g. if the residual MLP behaviour is
the suspected cause, search_docs("residual MLP standardisation") and
quote the relevant section rather than improvising.
```

### 2.9.7 Capacity & cost sizing

Order-of-magnitude numbers for your current corpus:

| Metric | Value |
| --- | --- |
| Documents indexed | ~150 |
| Total tokens | ~500k–800k (markdown is sparse) |
| Total chunks | ~1,500 at 800-token chunks |
| Full-rebuild embedding cost | ~$0.02 (text-embedding-3-small) |
| Per-query embedding cost | ~$0.000002 |
| Per-query retrieval latency | <50ms (in-memory chromadb) |
| Index storage | ~10 MB on disk |

A full rebuild costs less than one cup of coffee, and incremental updates are free at this scale.

---

# Part 3 — Capabilities Catalogue

## 3.1 Stakeholder personas

The capabilities below are labelled by which stakeholder gets the most value. The four personas in our environment:

| Persona | Role | What they care about | When they use the dashboard |
| --- | --- | --- | --- |
| **Trader** | Front office, takes risk | Real-time PnL, position changes, risk attribution, scenario sensitivities | Intraday — needs fast answers, low patience for clicking |
| **FO (Front Office) Analyst** | Quant support to traders | Trade-level explanations, what-if scenarios, structuring help | Throughout the day — supports trader's questions |
| **Risk Management** | Independent risk function, oversight | Drift, model breaks, governance, regulatory compliance, capital impact | Daily / weekly — methodical, needs evidence trails |
| **Quant** | Builds and maintains models | Model performance debugging, retraining decisions, scenario design | Throughout development cycles + when issues escalate |

Some capabilities are universal; others are clearly aimed at one persona. The catalogue makes this explicit.

## 3.2 Tier 1 — Conversational data access

Replaces clicking through tabs with a sentence. Low risk, fast to build, immediate value.

### T1.1 — NLQ for KPIs and metrics

> **"What is the portfolio MAE on the test set for the current ensemble?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | All (Trader, FO, Risk, Quant) |
| **Value** | Replaces remembering which tab → opening it → finding the KPI card |
| **Tools required** | `get_overview`, `get_portfolio_metrics` |
| **System prompt** | Standard chat prompt |
| **UI rendering** | Plain text + KPI mini-card |
| **Effort** | S (1 day) — part of Phase A |
| **Risk** | Low — pure read |

### T1.2 — Cluster ranking and filtering

> **"Show me the 5 worst-performing FX clusters on test"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk |
| **Value** | Today requires sorting a 50-row table and visually filtering by attribute |
| **Tools required** | `list_clusters` (with sort + filter params), `get_cluster_metrics` |
| **System prompt** | Tell LLM that cluster attributes include `desk`, `ccy`, `product_class` |
| **UI rendering** | Inline table with cluster cards; "Open in dashboard" link per row |
| **Effort** | S (1 day) |
| **Risk** | Low |

### T1.3 — Run history and lookup

> **"What inference runs did we do yesterday?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk (audit), Quant (debugging) |
| **Value** | Replaces opening run list, filtering by date |
| **Tools required** | `list_inference_runs`, `list_monitoring_runs` |
| **System prompt** | Today's date injected in context |
| **UI rendering** | Inline table; clicking a run navigates to the run page |
| **Effort** | S (½ day) |
| **Risk** | Low |

### T1.4 — Trade lookup

> **"Show me trade T-91823's prediction breakdown"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Trader (primary), FO |
| **Value** | Currently requires knowing which cluster the trade is in |
| **Tools required** | `lookup_trade`, `get_trade_predictions`, `get_trade_attributes` |
| **System prompt** | Standard |
| **UI rendering** | Trade card + small predicted-vs-target chart |
| **Effort** | S (1 day) |
| **Risk** | Low |

### T1.5 — Quick metric comparisons

> **"How does v2026-05-30 compare to v2026-05-23 on portfolio metrics?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk |
| **Value** | Replaces opening two governance tabs and copying numbers |
| **Tools required** | `get_portfolio_metrics` × 2 with different version params |
| **UI rendering** | Side-by-side KPI cards with deltas |
| **Effort** | S (1 day) |
| **Risk** | Low |

### T1.6 — Drift status check

> **"Is anything drifting badly right now?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk (primary), Quant |
| **Value** | Replaces opening monitoring tab → loading latest run → reading severity |
| **Tools required** | `get_latest_monitoring_run`, `get_drift_summary` |
| **UI rendering** | Plain text + severity badge + small histogram |
| **Effort** | S (½ day) |
| **Risk** | Low |

### T1.7 — Methodology Q&A (RAG-backed)

> **"How does our residual MLP work?"** / **"What's our PSI threshold methodology?"** / **"Why do we standardise PnL before training?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | All (onboarding for Quants, comfort for Risk, context for FO/Trader) |
| **Value** | Replaces "ask a senior quant" or "go read 5 markdown files" for any methodology question. Onboarding for new joiners drops from weeks to days. |
| **Tools required** | `search_docs` (RAG over project docs, see §1.9 and §2.9) |
| **System prompt** | "When a methodology question is asked, call search_docs first; cite every claim as `[<doc>, §<section>]`; refuse to improvise if chunks don't cover the question" |
| **UI rendering** | AI bubble with prose answer + citation chips at end of each paragraph; chip click opens the source doc at the right anchor |
| **Effort** | M (3 days — most of the lift is in §2.9 plumbing, then this capability is free) |
| **Risk** | Low — read-only, well-cited |

**Tier 1 cumulative effort: ~8 days** (was 5 days; +3 for the RAG plumbing). After this, ~85% of conversational *and* methodology questions are answered.

## 3.3 Tier 2 — Cross-source synthesis

Joining data from multiple endpoints to answer questions that today require switching tabs.

### T2.1 — Portfolio PnL attribution

> **"Why is portfolio PnL up 2% versus last week?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Trader (primary), FO, Risk |
| **Value** | Currently a manual exercise across inference + governance + cluster tabs |
| **Tools required** | `get_inference_run` × 2, `list_clusters_with_metrics`, `get_cluster_predictions` (top N contributors) |
| **System prompt** | Reasoning prompt asking for ranked attribution |
| **UI rendering** | Waterfall chart + bullet attribution + suggested follow-up actions |
| **Effort** | M (2 days) |
| **Risk** | Low — read-only synthesis |

### T2.2 — Cluster underperformance investigation

> **"Why is cluster cl-7 underperforming this week?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk |
| **Value** | Joins residuals + training curve + drift + trade composition; today 4 tabs of clicking |
| **Tools required** | `get_cluster_metrics`, `get_cluster_residuals`, `get_cluster_training_curve`, `get_cluster_drift`, `get_cluster_trade_composition` |
| **System prompt** | Reasoning prompt: rank 3 candidate causes with evidence strength |
| **UI rendering** | Multi-card output: each candidate cause = 1 card with chart + evidence |
| **Effort** | M (3 days) |
| **Risk** | Medium — hypotheses may be wrong; needs confidence indicator |

### T2.3 — Ensemble version comparison

> **"Compare ensemble v2026-05-30 with v2026-05-23 on the test set"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk |
| **Value** | Currently requires opening governance tab, loading both versions, comparing manually |
| **Tools required** | `get_overview` × 2, `get_portfolio_metrics` × 2, `list_clusters_with_metrics` × 2 |
| **UI rendering** | Comparison cards + diff table for clusters with biggest changes |
| **Effort** | M (2 days) |
| **Risk** | Low |

### T2.4 — Risk committee briefing

> **"Prepare a risk committee briefing on this ensemble's model health"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary), Quant |
| **Value** | Currently 30-60 minutes of manual screen-grabbing and Word-doc writing |
| **Tools required** | `get_overview`, `get_portfolio_metrics`, `list_clusters_with_metrics`, `get_latest_drift_summary`, `get_recent_governance_events` |
| **System prompt** | Heavy structured prompt: 5 sections (Executive summary, Performance, Drift, Recent changes, Recommendations) |
| **UI rendering** | Structured briefing card with 5 collapsible sections + export-to-PDF button |
| **Effort** | M (3 days) |
| **Risk** | Medium — needs human review before sending to committee |

### T2.5 — Trade-to-cluster context

> **"For trade T-91823, what does the model think and why?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Trader (primary), FO |
| **Value** | Joins trade attributes + cluster context + recent residuals + nearby trades in graph |
| **Tools required** | `lookup_trade`, `get_cluster_summary`, `get_trade_neighbours` (graph), `get_trade_predictions` |
| **UI rendering** | Trade card → cluster card → graph mini-view → recent prediction chart |
| **Effort** | M (3 days) |
| **Risk** | Low |

### T2.6 — Cross-cluster correlation diagnosis

> **"Which clusters tend to move together this week?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk |
| **Value** | Identifies hidden risk concentrations not visible from per-cluster views |
| **Tools required** | `get_group_correlations`, `get_cluster_predictions` (bulk) |
| **UI rendering** | Correlation matrix heatmap + narrative explanation |
| **Effort** | M (2 days) |
| **Risk** | Low |

**Tier 2 cumulative effort: +15 days**. After this, the AI is "useful for actual work" not just a toy.

## 3.4 Tier 3 — Reasoning and explanation

The AI goes beyond retrieval to produce hypotheses, attributions, and quantitative reasoning. **Requires structured outputs and evidence-citing system prompts**.

### T3.1 — Quantitative attribution

> **"Predicted PnL is $2.3M, target PnL is $1.8M. Explain the $0.5M gap."**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Trader (primary), Quant, Risk |
| **Value** | Quantifies what's driving prediction error — replaces a quant spending an hour |
| **Tools required** | `get_inference_run`, `get_cluster_predictions` (all), `get_cluster_residuals`, `decompose_pnl_gap` (NEW tool) |
| **System prompt** | Structured: produce `Attribution(total_gap, contributors=[Contributor(cluster, amount, residual_z_score), ...])` |
| **UI rendering** | Waterfall chart + ranked contributor cards with z-score badges |
| **Effort** | L (4 days) — needs the decompose_pnl_gap helper |
| **Risk** | Medium — quantitative claims; needs strict tool-citation requirement |

### T3.2 — Anomaly attribution chain

> **"Drift is critical on this run — what's the root cause?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary), Quant |
| **Value** | Replaces 30+ mins of investigating which risk factor → which clusters → why |
| **Tools required** | `get_drift_summary`, `get_cluster_drift_table` (all clusters), `lookup_rf_by_cluster`, `get_baseline_stats` |
| **System prompt** | Reasoning prompt: trace cause from risk factor → cluster impact → recommended action |
| **UI rendering** | Cause-chain diagram (RF → Clusters → Impact) + recommended actions |
| **Effort** | L (5 days) |
| **Risk** | Medium |

### T3.3 — Counter-factual reasoning

> **"If you exclude the 5 most-drifted clusters, what does portfolio MAE become?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk |
| **Value** | Lets users explore "what if" without rerunning anything |
| **Tools required** | `get_cluster_metrics` (all), `compute_filtered_portfolio_metric` (NEW pure-compute tool) |
| **System prompt** | "Use compute_filtered_portfolio_metric to answer counter-factual questions. Always show the original and the filtered side-by-side." |
| **UI rendering** | Side-by-side KPI cards with delta |
| **Effort** | M (3 days) |
| **Risk** | Low — pure compute, results are deterministic |

### T3.4 — Trend extraction and trajectory analysis

> **"Has cl-7's performance degraded steadily or suddenly?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk |
| **Value** | Identifies whether to retrain (sudden) or investigate data (gradual) |
| **Tools required** | `get_cluster_metrics_history` (NEW endpoint, reads recent inference runs) |
| **System prompt** | Reasoning prompt: classify trajectory (monotonic, sudden-shift, oscillating) and recommend action |
| **UI rendering** | Time-series chart with annotation + recommendation card |
| **Effort** | L (4 days) — needs the metrics history endpoint |
| **Risk** | Medium |

### T3.5 — Statistical significance reasoning

> **"Is the difference in MAE between v2026-05-30 and v2026-05-23 statistically meaningful?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk |
| **Value** | Prevents over-reading noise; supports promotion / rollback decisions |
| **Tools required** | `get_cluster_residuals` × 2 (paired), `compute_paired_t_test` (NEW pure-compute tool) |
| **System prompt** | "Report p-value, effect size, and a plain-English interpretation." |
| **UI rendering** | Statistical result card with confidence indicator |
| **Effort** | M (2 days) |
| **Risk** | Low |

### T3.6 — Hypothesis ranking

> **"Why might cluster cl-12 have started predicting too low?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary) |
| **Value** | Generates ranked hypothesis list with falsifiable tests |
| **Tools required** | Same as T2.2 + `list_recent_trade_changes`, `get_recent_market_moves` |
| **System prompt** | Structured: produce `HypothesisList(hypotheses=[Hypothesis(claim, evidence, test_to_falsify), ...])` ranked by likelihood |
| **UI rendering** | Hypothesis cards with strength bars + suggested test buttons |
| **Effort** | L (4 days) |
| **Risk** | Medium |

**Tier 3 cumulative effort: +22 days**. This is where the AI becomes a "thinking partner" not just a search bar.

## 3.5 Tier 4 — Generative actions

The AI produces draft artefacts the user reviews before they take effect. Requires write-capable tools + approval flows.

### T4.1 — Synthetic scenario generation

> **"Build me a stress scenario based on Brexit 2016, scaled 1.5x"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | FO (primary), Risk, Quant |
| **Value** | Currently requires manually editing shock CSVs by hand |
| **Tools required** | `list_historical_events`, `get_event_market_moves`, `draft_scenario` (NEW write tool — produces files in a staging dir) |
| **System prompt** | "Scenarios must specify shock values for every risk factor in the user's portfolio. Use historical-analogue scaling for any RFs not directly affected by the event." |
| **UI rendering** | Scenario preview table + "Apply to monitoring run" button (with approval modal) |
| **Effort** | L (5 days) |
| **Risk** | Medium — write capability needs human-in-the-loop |

### T4.2 — Regulator-facing explanation drafts

> **"Draft a paragraph explaining cluster cl-7's PSI spike for the FRTB regulator"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary) |
| **Value** | Replaces 30-60 mins of careful writing |
| **Tools required** | Same as T3.2 + `get_regulatory_template` |
| **System prompt** | Heavy: tone, vocabulary constraints, mandatory citation format, no hedging |
| **UI rendering** | Draft editor with citation chips + "Send to compliance review" button |
| **Effort** | L (4 days) |
| **Risk** | High — regulatory output; mandatory human review enforced in UI |

### T4.3 — Hyperparameter suggestions

> **"Suggest training hyperparameters for retraining cl-7 given its current residual pattern"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary) |
| **Value** | Encodes tuning intuition; helps junior quants |
| **Tools required** | `get_cluster_residuals`, `get_current_training_config`, `propose_training_config` (NEW pure compute) |
| **System prompt** | "Reason from residual patterns to specific hyperparameter changes. Justify each change." |
| **UI rendering** | Config diff view + "Apply to next training run" button (gated) |
| **Effort** | L (4 days) |
| **Risk** | Medium |

### T4.4 — Monitoring alert configuration

> **"Set up an alert when cluster cl-9's PSI exceeds 0.2"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary) |
| **Value** | Self-service alerting without engineering work |
| **Tools required** | `create_alert_rule` (NEW write tool) |
| **System prompt** | "Parse the user's intent into an AlertRule(metric, threshold, comparison, channel)" |
| **UI rendering** | Rule preview card + "Create alert" button (approval modal) |
| **Effort** | M (3 days) |
| **Risk** | Medium |

### T4.5 — Cluster grouping suggestions

> **"Group these 50 trades into 4 reasonable clusters for retraining"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary) |
| **Value** | Speeds up cluster design — currently a manual process |
| **Tools required** | `get_trade_attributes` (bulk), `propose_clustering` (NEW pure compute, uses simple feature distance) |
| **System prompt** | "Propose clusterings that group by economically meaningful attributes (currency, tenor, product type)." |
| **UI rendering** | Proposed clustering with editable group assignments |
| **Effort** | L (5 days) |
| **Risk** | Low |

### T4.6 — Report template instantiation

> **"Generate a weekly model health report"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary), Quant |
| **Value** | Currently takes ~1 hour weekly; AI does 90% of it |
| **Tools required** | Same as T2.4 + `generate_pdf_from_template` |
| **System prompt** | Template-bound; AI fills narrative slots only |
| **UI rendering** | Live PDF preview + section-by-section approval + Export |
| **Effort** | L (5 days) |
| **Risk** | Medium |

**Tier 4 cumulative effort: +26 days**. This is where the AI starts to materially save time on recurring work.

## 3.6 Tier 5 — Proactive and agentic

The AI takes action without being explicitly asked. **Highest WOW factor but requires background workers and careful triggering.**

### T5.1 — Morning briefing

> **Auto-generated at 7am: "Here's what changed overnight"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | All (especially Trader and Risk Management) |
| **Value** | Replaces opening 5 tabs to see what happened overnight |
| **Tools required** | All Tier 1 + Tier 2 tools |
| **Triggering** | Background `apscheduler` job at user-configured time |
| **Delivery** | (a) Notification badge on AI panel pill (b) Email digest (c) Slack/Teams webhook |
| **UI rendering** | Briefing card on landing dashboard with 5 sections; clicking each opens AI panel with full context |
| **Effort** | L (5 days) — needs background runner |
| **Risk** | Medium — emails/Slack need rate limiting and opt-out |

### T5.2 — Auto-investigation on alert

> **When monitoring detects critical drift, AI pre-investigates so users open the panel and find the answer waiting**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary), Quant |
| **Value** | Time-to-first-explanation drops from 30+ mins to ~0 |
| **Tools required** | T3.2 tools |
| **Triggering** | Hook into `infer_events` — when stage="monitoring", phase="severity_critical", invoke `/ai/investigate/{run_id}` in background |
| **Delivery** | Notification badge + pre-filled investigation in AI panel |
| **UI rendering** | T3.2 structured investigation card, but appears unprompted |
| **Effort** | L (4 days) |
| **Risk** | Medium — false positives create noise |

### T5.3 — Tab-aware spotlight

> **As user opens each tab, AI flags 1–2 interesting things proactively**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | All |
| **Value** | "Did you notice?" — surfaces things users would otherwise miss |
| **Tools required** | Tab-specific Tier 2 tools |
| **Triggering** | URL change → debounced background call to `/ai/spotlight?tab=...&context=...` |
| **Delivery** | Subtle inline banner at top of each page; dismissable |
| **UI rendering** | One-line teaser + "Tell me more" → opens AI panel |
| **Effort** | L (5 days) |
| **Risk** | Medium — risk of becoming annoying; needs strong relevance filter |

### T5.4 — Conversational debugging

> **When an API call fails, AI sees the error + recent logs and explains in plain English**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Platform engineers |
| **Value** | Replaces Slack escalation to platform team for "why did this fail?" |
| **Tools required** | `get_recent_logs`, `get_failed_request_context`, `lookup_error_pattern` |
| **Triggering** | Toast notification on UI error includes "Ask AI" button |
| **UI rendering** | Error card → AI explanation card |
| **Effort** | L (5 days) |
| **Risk** | Low — read-only |

### T5.5 — Intent prediction

> **AI suggests next likely action based on navigation pattern**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | All |
| **Value** | Reduces clicks by anticipating |
| **Tools required** | `get_user_recent_actions` (NEW endpoint), tab-specific tools |
| **Triggering** | After 30s on a page, AI panel shows a single suggested action |
| **UI rendering** | Subtle suggestion chip near AI pill |
| **Effort** | M (3 days) |
| **Risk** | Medium — privacy implications (need to log user actions); opt-in |

### T5.6 — Cross-run pattern detection

> **AI notices "this drift run looks similar to one 3 weeks ago" and suggests the same fix worked then**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management, Quant |
| **Value** | Institutional memory — captures "we've seen this before" |
| **Tools required** | `embed_drift_run` (NEW — produces vector embedding), `search_similar_runs` (vector search) |
| **Triggering** | Either on-demand ("any similar runs?") or auto-detected during investigation |
| **UI rendering** | Linked runs card with similarity score |
| **Effort** | XL (10 days) — needs vector DB or embedding store |
| **Risk** | Medium |

**Tier 5 cumulative effort: +32 days**. This is the "we built something special" tier.

## 3.7 Tier 6 — Domain-differentiating capabilities

Specific to your codebase and data; not buildable by anyone without your stack. **These are the demos that convince a stakeholder this is uniquely yours.**

### T6.1 — Cluster lineage narrative

> **"Tell me the story of cluster cl-7 across all ensemble versions"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk Management |
| **Value** | Replaces digging through governance for version history |
| **Tools required** | `get_cluster_lineage` (NEW — joins versions, retraining events, performance over time) |
| **System prompt** | "Tell the cluster's story chronologically with performance arc highlighted." |
| **UI rendering** | Vertical timeline with retraining markers + performance line chart underneath |
| **Effort** | L (5 days) |
| **Risk** | Low |

### T6.2 — Scenario coverage gap analysis

> **"Does our test scenario set cover the magnitude of last week's market moves?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Risk Management (primary), Quant |
| **Value** | Identifies blind spots in stress coverage — critical for regulatory comfort |
| **Tools required** | `get_test_scenario_distribution` (per RF), `get_recent_market_moves`, `compute_coverage_gap` |
| **System prompt** | "Identify RFs where recent moves exceed test scenario extremes. Recommend new scenarios to add." |
| **UI rendering** | Per-RF coverage chart (test min/max as bars, recent moves as dots overlaid) |
| **Effort** | L (5 days) |
| **Risk** | Low |

### T6.3 — Model decision explainability

> **"Why did the model predict $X for trade T?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | FO (primary), Trader, Risk Management |
| **Value** | Massive value for FO desks; today this is "the model is a black box" |
| **Tools required** | `get_gnn_attention` (NEW hook into model.forward, returns attention weights), `get_rnn_regime_classification` (NEW), `get_residual_decomposition` (NEW) |
| **System prompt** | Heavy: combine three internal signals into one coherent explanation |
| **UI rendering** | Trade card → 3 evidence cards (graph attention heatmap, RNN regime card, residual contribution) → synthesised paragraph |
| **Effort** | XL (10 days) — needs model hooks |
| **Risk** | Medium |

### T6.4 — Risk attribution debate

> **User: "I think cluster cl-12 is broken." AI: "Here's evidence for and against"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Risk Management |
| **Value** | Prevents premature retraining / disabling; provides balanced view |
| **Tools required** | Same as T2.2 + T3.5 |
| **System prompt** | "Argue both sides. Conclude with a recommended action and a confidence level." |
| **UI rendering** | Two-column "for / against" card + recommendation banner |
| **Effort** | L (5 days) |
| **Risk** | Medium — model risk: AI might be too confident |

### T6.5 — Auto-generate eval narratives for model-risk review

> **"Draft the model documentation paragraph for cluster cl-7"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk Management |
| **Value** | Replaces hours of writing per cluster for MRM submissions |
| **Tools required** | All Tier 1/2 tools, `get_cluster_governance_history`, `get_model_risk_template` |
| **System prompt** | Strict format, regulatory tone, no hedging, mandatory citations |
| **UI rendering** | Draft editor with section-by-section approve + export to MRM portal |
| **Effort** | XL (10 days) |
| **Risk** | High — regulatory artefact; mandatory review enforced |

### T6.6 — Pricing kernel anomaly detection

> **AI watches per-cluster pricing residuals and flags when the *pricing* (not the model) is drifting**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant, Trader |
| **Value** | Identifies when the upstream pricing library (rade_sr) has changed behaviour, not the model |
| **Tools required** | `get_pricing_residuals_trend`, `compare_pricing_versions` |
| **Triggering** | Background job comparing today's pricing kernel outputs to a reference set |
| **UI rendering** | Alert card with deviated trades highlighted |
| **Effort** | XL (8 days) |
| **Risk** | Medium |

### T6.7 — Documentation drift detection (RAG-enabled)

> **"The PSI threshold in code is 0.15 but the methodology doc still says 0.10. Which is correct, and when did this diverge?"**

| Aspect | Detail |
| --- | --- |
| **Stakeholders** | Quant (primary), Risk Management |
| **Value** | Catches documentation that no longer matches the code — a major audit/regulatory risk. The AI cross-checks key constants and method choices between code and documented methodology. |
| **Tools required** | `search_docs` (for documented value), `extract_constant_from_code` (NEW, AST-based), `git_blame_constant` (NEW, finds when the code value last changed) |
| **System prompt** | Heavy: must compare doc value vs code value vs git history and flag discrepancies as either *doc-stale* (code is right) or *code-drift* (doc is right) or *ambiguous* |
| **Triggering** | (a) on-demand "audit cl-7's methodology vs code", (b) background nightly sweep over a curated list of "important constants" with notification on new mismatches |
| **UI rendering** | Mismatch card with three columns: doc value + code value + last-changed-when |
| **Effort** | L (6 days) |
| **Risk** | Medium — false positives possible if the AI misinterprets either side |

## 3.8 Prioritisation matrix

Plotting effort vs. value (qualitative; calibrate as you go):

```
            ▲ Stakeholder value
       HIGH │
            │
            │  ┌──────┐                      ┌──────┐
            │  │ T3.1 │              T6.3 →  │ MODEL│
            │  │ ATTR │                      │  XAI │
            │  └──────┘                      └──────┘
            │  ┌──────┐  ┌──────┐                    ┌──────┐
            │  │ T2.2 │  │ T2.4 │                    │ T6.5 │
            │  │ CLUS │  │ COMM │                    │ MRM  │
            │  │ DBG  │  │ BRIEF│                    │ DOC  │
            │  └──────┘  └──────┘                    └──────┘
            │  ┌──────┐  ┌──────┐  ┌──────┐
            │  │ T5.1 │  │ T5.2 │  │ T3.2 │
            │  │ DAILY│  │ AUTO │  │ DRIFT│
            │  │ BRIEF│  │ INV  │  │ CAUSE│
            │  └──────┘  └──────┘  └──────┘
       MED  │
            │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
            │  │ T1.1 │  │ T1.2 │  │ T4.6 │  │ T6.1 │
            │  │ KPI  │  │ RANK │  │ REPRT│  │ LINE │
            │  │ LKP  │  │      │  │      │  │ AGE  │
            │  └──────┘  └──────┘  └──────┘  └──────┘
            │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
            │  │ T1.3 │  │ T1.6 │  │ T2.3 │  │ T4.1 │
            │  │ RUN  │  │ DRIFT│  │ VER  │  │ SCEN │
            │  │ LIST │  │ CHK  │  │ DIFF │  │ GEN  │
            │  └──────┘  └──────┘  └──────┘  └──────┘
       LOW  │
            └────────────────────────────────────────────────► Effort
                S            M            L            XL
```

**Recommended ordering**:

1. **Quick wins** (high value, S/M effort): T1.1, T1.2, T1.6, T2.2, T2.4 — ship first ~4 weeks
2. **Differentiators** (high value, L effort): T3.1, T3.2, T5.1, T5.2 — ship in months 2–3
3. **Transformers** (massive value, XL effort): T6.3, T6.5 — ship later, drives strategic conversations

---

# Part 4 — Mock Visualisations

The following mocks illustrate selected capabilities in action. They are stylistically consistent with the existing `rade_*.png` family (dark navy, violet accent, card-based layout, AI panel docked to the right).

## Mock 1 — Tier 1: Conversational data access ("worst clusters this week")

> User asks for a ranked list; AI returns a compact bar chart inline plus a clickable cluster table. Replaces sorting a large evaluation table by hand.

![Tier 1 NLQ ranking](./platform_designs/rade_ai_mock_t1_worst_clusters.png)

## Mock 2 — Tier 1: NLQ filter ("MAPE for FX clusters")

> Free-text filter against the cluster catalog; AI returns a focused bar chart and table without the user navigating to evaluation.

![Tier 1 NLQ filter](./platform_designs/rade_ai_mock_t1_fx_mape.png)

## Mock 3 — Tier 2: Cross-source investigation ("why is cl-7 underperforming?")

> AI joins residuals, training curves, drift status, and trade composition — producing 3 ranked candidate causes, each with its own evidence card and a chart.

![Tier 2 cluster investigation](./platform_designs/rade_ai_mock_t2_cluster_investigate.png)

## Mock 4 — Tier 3: Quantitative attribution ("explain the $0.5M PnL gap")

> AI decomposes a portfolio gap into ranked cluster contributors with z-scores, evidence chips, and a waterfall chart — replacing an hour of quant analysis.

![Tier 3 quant attribution](./platform_designs/rade_ai_mock_t3_quant_attribution.png)

## Mock 5 — Tier 5: Morning briefing (proactive)

> No question asked — AI runs overnight and posts a 5-section briefing to the landing dashboard. Each section has one chart and clear "what changed" deltas.

![Tier 5 morning briefing](./platform_designs/rade_ai_mock_t5_morning_briefing.png)

## Mock 6 — Tier 5: Auto-investigation on alert (proactive)

> Monitoring drift goes critical → AI auto-investigates → user opens the panel and finds a pre-built investigation waiting, with cause chain and recommended actions.

![Tier 5 auto-investigation](./platform_designs/rade_ai_mock_t5_auto_investigate.png)

---

# Appendix A — OpenAI API specifics

The implementation assumes the OpenAI Python SDK (`openai>=1.50`) and either direct OpenAI access or a compatible enterprise gateway.

## Key SDK features to use

| Feature | Purpose | Endpoint / SDK |
| --- | --- | --- |
| **Chat Completions API** | Standard chat with tools + streaming | `client.chat.completions.create(..., stream=True)` |
| **Function calling (`tools` parameter)** | Lets the LLM call your dispatch handlers | `tools=[{type: "function", function: {...}}]` |
| **Structured outputs (`response_format`)** | Force the model to return JSON matching a Pydantic schema — critical for T3 quantitative answers and T4 drafts | `response_format={type: "json_schema", json_schema: {...}}` |
| **Streaming via SSE** | Token-by-token UX | iterate over `client.chat.completions.create(stream=True)` |
| **`tool_choice="auto" / "required" / "none"`** | Force / forbid tool use per turn | Useful for guard-railing certain endpoints |
| **`temperature`** | Set 0–0.3 for factual, 0.7 for creative drafts | per call |
| **`max_tokens`** | Cap response size; protect cost | per call |
| **`reasoning_effort`** (o-series models) | Trade speed for quality on hard reasoning tasks | per call, `low | medium | high` |
| **Batch API** | 50% discount for non-interactive workloads (morning briefings) | `client.batches.create(...)` |
| **Embeddings** | T5.6 cross-run similarity | `client.embeddings.create(input=..., model="text-embedding-3-small")` |

## Model selection guidelines

| Model | When to use | Cost (approx, 2026) |
| --- | --- | --- |
| `gpt-4o-mini` | Tier 1 lookups, simple tool calls | $0.15 / $0.60 per M tokens |
| `gpt-4o` | Tier 2 synthesis, Tier 3 reasoning | $2.50 / $10 per M tokens |
| `o1-mini` / `o3-mini` | Hard reasoning (T3.6, T6.3) | $1.10 / $4.40 per M tokens |
| `o1` / `o3` | Critical model-risk work where reasoning matters more than latency | $15 / $60 per M tokens |

## Streaming response chunk shapes

```python
# Text chunk
{
    "choices": [{
        "delta": {"content": "The worst-performing cluster is "},
        "index": 0,
    }]
}

# Tool call chunk (arrives at the start of a tool use)
{
    "choices": [{
        "delta": {
            "tool_calls": [{
                "index": 0,
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_cluster_metrics", "arguments": ""}
            }]
        }
    }]
}

# Subsequent chunks stream the arguments
{
    "choices": [{
        "delta": {
            "tool_calls": [{"index": 0, "function": {"arguments": '{"cluster_id": "cl-7"}'}}]
        }
    }]
}

# Done
{"choices": [{"finish_reason": "tool_calls" | "stop"}]}
```

The session manager accumulates `tool_calls[i].function.arguments` across chunks before parsing JSON, dispatching, and looping.

---

# Appendix B — Tool registry skeleton

The initial registry of 13 tools covers ~85% of conversational and methodology use cases.

```python
# src/rade_ml_pt/ensemble/api/services/ai_tools.py

from typing import Any, Callable, Dict, List
from pydantic import BaseModel

# Each tool maps directly to an existing service function.
# NO new business logic in this file — pure plumbing.

class ToolDef(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]    # JSON Schema (sent to LLM)
    handler: Callable                # callable(input_dict, ctx) -> tool_result
    mutates: bool = False            # write capability flag
    requires_approval: bool = False  # gates write tools


TOOLS: List[ToolDef] = [
    # ── Tier 1 lookups ─────────────────────────────────────────────
    ToolDef(
        name="get_ensemble_overview",
        description="Get headline information about the currently loaded ensemble model — version, "
                    "training date, cluster count, portfolio metrics.",
        input_schema={"type": "object", "properties": {}, "required": []},
        handler=lambda inp, ctx: ctx.overview_svc.get_overview(),
    ),
    ToolDef(
        name="list_clusters",
        description="List clusters with optional filtering by attribute (desk, currency, product_class). "
                    "Returns cluster_id, name, attributes, headline metric.",
        input_schema={
            "type": "object",
            "properties": {
                "desk":          {"type": "string"},
                "currency":      {"type": "string"},
                "product_class": {"type": "string"},
                "sort_by":       {"type": "string",
                                  "enum": ["mae_asc", "mae_desc", "rmse_asc", "rmse_desc"]},
                "limit":         {"type": "integer", "default": 50},
            },
            "required": [],
        },
        handler=lambda inp, ctx: ctx.cluster_svc.list_with_metrics(**inp),
    ),
    ToolDef(
        name="get_cluster_summary",
        description="Get full summary for one cluster: KPIs, attributes, recent residuals.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster_id": {"type": "string"},
                "split":      {"type": "string", "enum": ["train", "val", "test"], "default": "test"},
            },
            "required": ["cluster_id"],
        },
        handler=lambda inp, ctx: ctx.cluster_svc.get_summary(**inp),
    ),
    ToolDef(
        name="get_cluster_metrics",
        description="Numerical evaluation metrics (MAE, RMSE, MAPE, R²) for one cluster on a split.",
        input_schema={
            "type": "object",
            "properties": {
                "cluster_id": {"type": "string"},
                "split":      {"type": "string", "enum": ["train", "val", "test"]},
            },
            "required": ["cluster_id"],
        },
        handler=lambda inp, ctx: ctx.cluster_svc.get_metrics(**inp),
    ),
    ToolDef(
        name="get_portfolio_metrics",
        description="Aggregate portfolio-level metrics for an ensemble version.",
        input_schema={
            "type": "object",
            "properties": {
                "ensemble_version": {"type": "string", "description": "Defaults to current"},
                "split":            {"type": "string", "enum": ["train", "val", "test"]},
            },
            "required": [],
        },
        handler=lambda inp, ctx: ctx.portfolio_svc.get_metrics(**inp),
    ),
    ToolDef(
        name="list_inference_runs",
        description="List recent inference runs, newest first.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "since": {"type": "string", "description": "ISO date, e.g. '2026-06-01'"},
            },
            "required": [],
        },
        handler=lambda inp, ctx: ctx.inference_svc.list_runs(**inp),
    ),
    ToolDef(
        name="get_inference_run",
        description="Detailed view of one inference run: scenario count, cluster predictions, portfolio totals.",
        input_schema={
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
        handler=lambda inp, ctx: ctx.inference_svc.get_run(**inp),
    ),
    ToolDef(
        name="list_monitoring_runs",
        description="List recent drift monitoring runs, newest first.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 20},
                "severity": {"type": "string", "enum": ["info", "warn", "critical"]},
            },
            "required": [],
        },
        handler=lambda inp, ctx: ctx.monitoring_svc.list_runs(**inp),
    ),
    ToolDef(
        name="get_drift_summary",
        description="Portfolio-level drift summary for a monitoring run: severity, n_clusters affected, "
                    "top contributing risk factors.",
        input_schema={
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
        handler=lambda inp, ctx: ctx.monitoring_svc.get_drift_summary(**inp),
    ),
    ToolDef(
        name="get_cluster_drift",
        description="Per-feature drift table for one cluster on one monitoring run.",
        input_schema={
            "type": "object",
            "properties": {
                "run_id":     {"type": "string"},
                "cluster_id": {"type": "string"},
            },
            "required": ["run_id", "cluster_id"],
        },
        handler=lambda inp, ctx: ctx.monitoring_svc.get_cluster_drift(**inp),
    ),

    # ── Tier 2 visualisation tools ─────────────────────────────────
    ToolDef(
        name="render_chart",
        description="Render a chart inline in the chat. Provide a Plotly figure spec following "
                    "the JSON schema. Only the supported chart types are listed; do not invent others.",
        input_schema={
            "type": "object",
            "properties": {
                "chart_type": {"type": "string",
                               "enum": ["line", "bar", "scatter", "waterfall", "heatmap", "histogram"]},
                "title":      {"type": "string"},
                "data":       {"type": "object", "description": "Plotly-compatible data spec"},
                "layout":     {"type": "object", "description": "Plotly layout overrides (optional)"},
            },
            "required": ["chart_type", "data"],
        },
        handler=lambda inp, ctx: {"rendered": True, "spec": inp},
    ),
    ToolDef(
        name="render_table",
        description="Render a table inline in the chat with columns and rows.",
        input_schema={
            "type": "object",
            "properties": {
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows":    {"type": "array",
                            "items": {"type": "array", "items": {"type": ["string", "number", "null"]}}},
                "title":   {"type": "string"},
            },
            "required": ["columns", "rows"],
        },
        handler=lambda inp, ctx: {"rendered": True, "spec": inp},
    ),

    # ── Knowledge augmentation (RAG) ───────────────────────────────
    ToolDef(
        name="search_docs",
        description=(
            "Search project documentation (model methodology, framework guides, pipeline "
            "contracts, architecture docs) for explanatory content. Use when the user asks "
            "'how does X work?', 'why is X structured this way?', 'what is our methodology "
            "for X?', or any question about the system's design rather than its current "
            "state. Returns 3-5 most relevant chunks with citations of the form "
            "[<doc>, §<section>]. ALWAYS cite chunks when answering; refuse to improvise "
            "if the returned chunks do not cover the question."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query describing what to find."
                },
                "tier": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["model_methodology", "architecture",
                                 "framework_guide", "user_doc"]
                    },
                    "description": (
                        "Restrict search to one or more document tiers. "
                        "Use ['model_methodology', 'architecture'] for serious "
                        "methodology questions; add 'framework_guide' for broader "
                        "onboarding-style questions; add 'user_doc' for "
                        "'how do I use the dashboard' questions."
                    ),
                },
                "top_k": {"type": "integer", "default": 5, "maximum": 10},
            },
            "required": ["query"],
        },
        handler=lambda inp, ctx: ctx.doc_retriever.search(**inp),
    ),
]


def dispatch(tool_name: str, tool_input: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
    """Invoke a registered tool. Returns a dict that will be sent back to the LLM."""
    tool = _by_name(tool_name)
    if tool.mutates and not ctx.approval_granted:
        return {"error": "Tool requires approval; ask the user to confirm."}
    try:
        result = tool.handler(tool_input, ctx)
        return {"ok": True, "data": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _by_name(name: str) -> ToolDef:
    for tool in TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(f"Tool not registered: {name}")


def openai_schema() -> List[Dict[str, Any]]:
    """Convert TOOLS into OpenAI's `tools` request format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in TOOLS
    ]
```

---

# Appendix C — System prompt templates

System prompts shape behaviour more than any other lever. Three template shapes cover most needs.

## C.1 — Base chat prompt (Tier 1 / 2)

```
You are PRISM AI, embedded in the RADE analytics dashboard for an ensemble of
cluster-level hybrid GNN-RNN models that predict portfolio P&L under stress
scenarios.

CONTEXT:
  - User: {user_id}
  - Current page: {tab_name}
  - Active ensemble version: {ensemble_version}
  - Selected cluster (if any): {cluster_id}
  - Selected monitoring run (if any): {run_id}
  - Date today: {iso_date}

INSTRUCTIONS:
  - Answer questions using the tools provided. Do NOT invent numbers.
  - When the user uses "this", "the cluster", "the run", interpret as the
    active selection above unless they specify otherwise.
  - For numerical answers, always cite the tool call that produced them by
    including the cluster_id, run_id, and metric name inline.
  - Prefer rendering tables and charts over long prose for data-heavy answers.
  - Use `render_chart` only with chart types listed in its schema.
  - If a question cannot be answered with available tools, say so plainly.
  - Keep responses concise; finance professionals value brevity.
```

## C.2 — Investigation prompt (Tier 3)

```
You are PRISM Investigator. The user (or the system) has asked you to
investigate why {target} is exhibiting {symptom}. Produce a structured
investigation report following the JSON schema provided.

PROCESS:
  1. Gather evidence using the tools provided.
  2. Form 2-4 candidate hypotheses ranked by evidence strength.
  3. For each hypothesis, cite the specific tool call and number that supports
     or contradicts it.
  4. Choose the most likely hypothesis and recommend a specific next action.

OUTPUT MUST FOLLOW THE SCHEMA EXACTLY. Required fields:
  - summary: one-paragraph executive summary
  - hypotheses: list of {claim, evidence_for, evidence_against, likelihood}
  - recommended_action: one of {investigate_further, retrain, accept, escalate}
  - confidence: low | medium | high
  - evidence_charts: list of chart specs (renderable inline)

CONSTRAINTS:
  - Do NOT speculate beyond evidence. If a hypothesis is unsupported, say "no
    evidence available" rather than guessing.
  - Confidence = "high" requires at least 2 independent tools supporting the
    conclusion.
  - If you cannot reach a confident conclusion, set confidence="low" and
    recommend investigate_further with specific next steps.
```

## C.3 — Draft/generation prompt (Tier 4)

```
You are PRISM Drafter. The user wants you to draft {artefact_type}.

ROLE: producing first-draft material for HUMAN REVIEW. You are not the final
author; a domain expert will edit before any external publication.

CONSTRAINTS:
  - Use {tone_register}: formal | conversational | regulatory
  - Maximum length: {word_limit} words
  - Cite all numerical claims using the tools provided
  - Do not include hedging language ("seems", "appears to") in regulatory artefacts
  - Mark every draft section with [REVIEW REQUIRED] tags around any claim that
    could not be verified by a tool call

OUTPUT FORMAT: structured per the schema. The user will see each section
separately with edit-and-approve controls.

If the request requires data you cannot fetch, return a draft with [MISSING:
<what you would need>] placeholders rather than guessing.
```

---

# Appendix D — Implementation phase plan

## Phase A — Plumbing & hello-world chat (Week 1)

**Goal**: end-to-end "I can chat with the LLM in the dashboard."

- [ ] Add `openai>=1.50` to dependencies
- [ ] Implement `ai_client.py` (sync chat, no tools, no streaming)
- [ ] Implement `ai_audit.py` (sqlite-backed)
- [ ] Add `ai_audit_log` + `ai_conversations` tables to `db_schema.sql`
- [ ] Implement `routers/ai.py` with `POST /ai/chat` (single-turn, sync)
- [ ] Implement `models/ai.py` (ChatRequest, ChatResponse)
- [ ] Wire `ai_router` into `app.py`; add AI dependencies
- [ ] Implement `layouts/ai_panel.py` (floating pill + drawer + simple chat thread)
- [ ] Implement `callbacks/ai_assistant_cb.py` (sync send + receive)
- [ ] Add `chat_with_ai()` to `api_client.py`
- [ ] Mount panel in app shell
- [ ] **Acceptance**: type "say hello" → get response → audit log row exists

**Effort**: 4–5 days. **Risk**: confirming work's OpenAI endpoint actually works programmatically.

## Phase B — Tool calling (Weeks 2–3)

**Goal**: AI can answer real data questions using existing endpoints.

- [ ] Implement `ai_tools.py` with the 12 tools from Appendix B
- [ ] Implement `ai_session.py` (multi-turn loop, in-memory conversation store)
- [ ] Extend `routers/ai.py` to use the session manager
- [ ] Add OpenAI `tools` parameter; handle `tool_calls` in response
- [ ] Update `models/ai.py` with tool_use / tool_result types
- [ ] Update UI to render tool_use steps as collapsible "Looked at X" chips
- [ ] **Acceptance**: ask "what's the test MAE on cl-7?" → get real number

**Effort**: 6–8 days. **Risk**: tool dispatch edge cases (timeouts, exceptions, large results).

### Phase B+ — Knowledge augmentation (RAG over docs, optional within Week 3)

**Goal**: AI can answer methodology / "how does it work?" questions citing project docs (T1.7).

- [ ] Add `chromadb` + `openai` embeddings (or local `sentence-transformers`) to dependencies
- [ ] Implement `ai_doc_index.py` with markdown-aware chunker
- [ ] Implement `ai_doc_retriever.py`
- [ ] Implement `scripts/reindex_docs.py` CLI
- [ ] Register `search_docs` tool in `ai_tools.py`
- [ ] Extend system prompt with knowledge-tool guidance (Appendix C C.1)
- [ ] Configure doc roots + tier mapping in `Settings`
- [ ] Initial bulk index of `docs/`, READMEs, and `paper/` LaTeX
- [ ] Wire `apscheduler` background re-index every 5 minutes
- [ ] Add `GET /ai/docs/health` and `GET /ai/docs/stats` endpoints
- [ ] **Acceptance**: ask "how does our residual MLP work?" → answer with `[<doc>, §<section>]` citations that resolve to the right line in the source markdown

**Effort**: 3 days. **Risk**: chunking quality — iterate on chunk size if recall is poor.

## Phase C — Streaming + inline charts (Week 4)

**Goal**: the panel feels like ChatGPT.

- [ ] Add `POST /ai/chat/stream` SSE endpoint
- [ ] Convert `ai_session.run_chat` to async generator
- [ ] Implement `render_chart` and `render_table` tools
- [ ] Implement clientside callback using `EventSource` for streaming
- [ ] Implement inline Plotly figure rendering in message bubbles
- [ ] Add structured outputs for chart specs (Plotly JSON schema)
- [ ] **Acceptance**: ask "plot residuals for cl-7" → response streams with chart at the end

**Effort**: 4–5 days. **Risk**: SSE through any reverse proxies (test the production deployment).

## Phase D — Context awareness (Week 5)

**Goal**: "this", "the cluster", "the run" just work.

- [ ] Add `page_context` field to `ChatRequest`
- [ ] Update Dash callback to inject current tab, ensemble version, selected cluster/run
- [ ] Update system prompt template to interpolate context
- [ ] Add context chips to AI panel header
- [ ] Add per-tab suggested prompts (`AI_QUICK_PROMPTS_BY_TAB`)
- [ ] **Acceptance**: navigate to cluster cl-7 detail → ask "explain this" → AI knows what "this" means

**Effort**: 2–3 days. **Risk**: low.

## Phase E — Specialised agents (Weeks 6–7)

**Goal**: structured investigations and reports.

- [ ] Implement Investigation schema (Pydantic with structured outputs)
- [ ] Implement `POST /ai/investigate/{monitoring_run_id}` (T3.2 / Mock 4-style)
- [ ] Implement `POST /ai/summarise/inference/{inference_run_id}`
- [ ] Implement `POST /ai/generate-report` with template library
- [ ] Implement `components/ai_investigation_card.py`
- [ ] Implement PDF export via WeasyPrint
- [ ] **Acceptance**: click "investigate" on a monitoring run → structured 5-section report

**Effort**: 8–10 days. **Risk**: prompt engineering iteration time.

## Phase F — Proactive surface (Weeks 8–10)

**Goal**: AI starts acting without being asked.

- [ ] Add `apscheduler` for background jobs
- [ ] Implement morning briefing job (`/ai/briefing/daily`)
- [ ] Add notification badge to AI pill
- [ ] Hook into `infer_events`: critical drift → auto-invoke `/ai/investigate/{run_id}`
- [ ] Implement opt-in scheduler UI in user settings
- [ ] Implement T5.4 conversational debugging hook on errors
- [ ] **Acceptance**: critical drift in monitoring → 30 seconds later, notification badge shows; click → investigation waiting

**Effort**: 10–12 days. **Risk**: false positives create noise; rate-limit carefully.

## Phase G — Domain capabilities (Weeks 11+)

**Goal**: the wow-factor capabilities specific to your stack.

Build T6 capabilities individually as appetite + stakeholder priority dictates. Each is its own mini-project. Start with T6.3 (Model decision explainability) or T6.5 (MRM doc auto-draft) based on which stakeholder is pushing hardest.

## Cumulative timeline

| Phase | Weeks | Capabilities delivered | Cumulative effort |
| --- | --- | --- | --- |
| A | 1 | Hello-world chat | ~5 dev days |
| B | 2–3 | Tier 1 (all 6 caps) | ~13 dev days |
| **B+** | **3** | **RAG + T1.7 methodology Q&A** | **~16 dev days** |
| C | 4 | Streaming + inline charts | ~21 dev days |
| D | 5 | Tier 2.1, 2.2, 2.3 + context | ~24 dev days |
| E | 6–7 | Tier 2.4–6, Tier 3.1–2, Tier 4.6 | ~34 dev days |
| F | 8–10 | Tier 5.1, 5.2 + parts of 5.3–4 | ~46 dev days |
| G | 11+ | Tier 6 (one at a time, incl. T6.7) | open-ended |

Roughly **3 months for Phases A–F**, after which the AI assistant is a genuine differentiator. Phase G is "best in class" territory.

---

# Appendix E — RAG ingestion pipeline reference

Reference implementation for §1.9 + §2.9. ~150 lines covering chunking, embedding, persistence, and retrieval. Drop-in compatible with the `search_docs` tool from Appendix B.

## E.1 — `ai_doc_index.py`

```python
# src/rade_ml_pt/ensemble/api/services/ai_doc_index.py
"""Document indexer for RAG. Chunks markdown/LaTeX, embeds, persists to chromadb."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "rade_docs"


@dataclass
class DocChunk:
    """One indexed unit of documentation."""
    chunk_id:       str          # deterministic: f"{doc_hash}:{section_path}:{ord}"
    text:           str
    doc_path:       str          # repo-relative
    section_chain:  str          # "Hybrid GNN-RNN › Residual MLP › Standardisation"
    tier:           str          # 'model_methodology' | 'architecture' | ...
    doc_hash:       str          # sha256 of source file at index time
    chunk_ord:      int          # position within doc
    tokens:         int
    metadata:       dict = field(default_factory=dict)


class DocIndexer:
    """Build and maintain the document index."""

    def __init__(
        self,
        store_path: str,
        embed_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        openai_client: Optional[OpenAI] = None,
        ingestion_log_path: Optional[str] = None,
    ):
        self.client = chromadb.PersistentClient(
            path=store_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(_COLLECTION_NAME)
        self.embed_model = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai = openai_client or OpenAI()
        self.log_path = Path(ingestion_log_path) if ingestion_log_path else None

    # ── public API ──────────────────────────────────────────────────────

    def ingest_directory(
        self,
        root: str | Path,
        tier: str,
        exclude_globs: Iterable[str] = (),
    ) -> int:
        """Index every markdown/LaTeX file under `root`. Returns total chunks indexed."""
        root = Path(root)
        n_chunks = 0
        for path in self._walk(root, exclude_globs):
            n_chunks += self.ingest_file(path, tier)
        return n_chunks

    def ingest_file(self, path: str | Path, tier: str) -> int:
        """Index a single file. Skip if hash unchanged. Returns chunks indexed."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Skip if unchanged
        existing = self.collection.get(where={"doc_path": str(path)}, limit=1)
        if existing and existing["metadatas"]:
            if existing["metadatas"][0].get("doc_hash") == doc_hash:
                return 0

        # Delete superseded
        self.collection.delete(where={"doc_path": str(path)})

        # Chunk + embed + upsert
        chunks = list(self._chunk_markdown(text, str(path), tier, doc_hash))
        if not chunks:
            return 0
        embeddings = self._embed_batch([c.text for c in chunks])
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{
                "doc_path": c.doc_path,
                "section_chain": c.section_chain,
                "tier": c.tier,
                "doc_hash": c.doc_hash,
                "chunk_ord": c.chunk_ord,
                "tokens": c.tokens,
            } for c in chunks],
        )
        self._log_ingestion(path, tier, len(chunks), doc_hash)
        return len(chunks)

    def delete_by_path(self, path: str | Path) -> None:
        self.collection.delete(where={"doc_path": str(path)})

    def stats(self) -> dict:
        return {
            "n_chunks": self.collection.count(),
            "model":    self.embed_model,
        }

    # ── internal ────────────────────────────────────────────────────────

    def _walk(self, root: Path, exclude_globs: Iterable[str]) -> Iterable[Path]:
        if root.is_file():
            yield root
            return
        excludes = list(exclude_globs)
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".md", ".tex"}:
                continue
            if any(path.match(g) for g in excludes):
                continue
            yield path

    def _chunk_markdown(
        self,
        text: str,
        doc_path: str,
        tier: str,
        doc_hash: str,
    ) -> Iterable[DocChunk]:
        """Markdown-aware chunker: splits on headers, packs sections into ~chunk_size tokens."""
        lines = text.splitlines()
        section_stack: List[str] = []
        buf: List[str] = []
        ord_ = 0

        def flush() -> Iterable[DocChunk]:
            nonlocal buf, ord_
            if not buf:
                return
            joined = "\n".join(buf).strip()
            if not joined:
                buf = []
                return
            section_chain = " › ".join(section_stack) or "(root)"
            for sub in self._split_to_size(joined):
                yield DocChunk(
                    chunk_id=f"{doc_hash[:12]}:{ord_:04d}",
                    text=sub,
                    doc_path=doc_path,
                    section_chain=section_chain,
                    tier=tier,
                    doc_hash=doc_hash,
                    chunk_ord=ord_,
                    tokens=_approx_tokens(sub),
                )
                ord_ += 1
            buf = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                yield from flush()
                level = len(stripped) - len(stripped.lstrip("#"))
                heading = stripped.lstrip("# ").strip()
                section_stack = section_stack[: level - 1] + [heading]
            else:
                buf.append(line)
        yield from flush()

    def _split_to_size(self, text: str) -> Iterable[str]:
        if _approx_tokens(text) <= self.chunk_size:
            yield text
            return
        # Approximate split by paragraph then by sentence
        paras = text.split("\n\n")
        cur: List[str] = []
        cur_tok = 0
        for p in paras:
            t = _approx_tokens(p)
            if cur_tok + t > self.chunk_size and cur:
                yield "\n\n".join(cur)
                # Overlap: keep last paragraph
                cur = [cur[-1], p] if self.chunk_overlap > 0 else [p]
                cur_tok = sum(_approx_tokens(x) for x in cur)
            else:
                cur.append(p)
                cur_tok += t
        if cur:
            yield "\n\n".join(cur)

    def _embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.openai.embeddings.create(model=self.embed_model, input=chunk)
            out.extend([d.embedding for d in resp.data])
        return out

    def _log_ingestion(self, path: Path, tier: str, n_chunks: int, doc_hash: str) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({
                "path": str(path),
                "tier": tier,
                "n_chunks": n_chunks,
                "doc_hash": doc_hash,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }) + "\n")


def _approx_tokens(text: str) -> int:
    """Cheap token estimate: ~4 chars per token for English."""
    return max(1, len(text) // 4)
```

## E.2 — `ai_doc_retriever.py`

```python
# src/rade_ml_pt/ensemble/api/services/ai_doc_retriever.py
"""Query-side: similarity search + citation formatting for the LLM."""

from __future__ import annotations

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI


class DocRetriever:
    def __init__(
        self,
        store_path: str,
        embed_model: str = "text-embedding-3-small",
        openai_client: Optional[OpenAI] = None,
    ):
        client = chromadb.PersistentClient(
            path=store_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = client.get_collection("rade_docs")
        self.embed_model = embed_model
        self.openai = openai_client or OpenAI()

    def search(
        self,
        query: str,
        top_k: int = 5,
        tier: Optional[List[str]] = None,
    ) -> dict:
        """Return chunks + LLM-ready formatted block."""
        qvec = self.openai.embeddings.create(
            model=self.embed_model, input=[query]
        ).data[0].embedding

        where = {"tier": {"$in": tier}} if tier else None
        result = self.collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            where=where,
        )
        chunks = self._unpack(result)
        return {
            "chunks":    chunks,
            "formatted": self.format_for_llm(chunks),
        }

    def format_for_llm(self, chunks: List[dict]) -> str:
        """Render chunks as a single string the LLM can read + cite."""
        blocks = []
        for c in chunks:
            citation = f"[{c['doc_path']}, §{c['section_chain']}]"
            blocks.append(f"{citation}\n{c['text']}\n")
        return "\n---\n".join(blocks)

    def _unpack(self, result: dict) -> List[dict]:
        out = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, _id in enumerate(ids):
            out.append({
                "chunk_id":      _id,
                "text":          docs[i],
                "doc_path":      metas[i]["doc_path"],
                "section_chain": metas[i]["section_chain"],
                "tier":          metas[i]["tier"],
                "similarity":    1.0 - dists[i],
            })
        return out
```

## E.3 — `scripts/reindex_docs.py` (CLI)

```python
# scripts/reindex_docs.py
"""CLI to (re)build the RAG index for rade_analytics."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from rade_ml_pt.ensemble.api.config import get_settings
from rade_ml_pt.ensemble.api.services.ai_doc_index import DocIndexer

logger = logging.getLogger("reindex_docs")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


_TIER_ROOTS = {
    "model_methodology": [
        "docs/reference/models",
        "docs/reference/machine_learning",
        "docs/reference/risk",
        "docs/rade_ml/RADE_ML_IMPLEMENTATION_PLAN.md",
        "docs/rade_ml_pt_implementation_guide.md",
        "docs/inference_pipeline_contract.md",
        "docs/paper/section_data_pipeline.tex",
        "docs/paper/section_data_pipeline_companion.md",
    ],
    "architecture": [
        "docs/platform_designs/RADE_UI_DESIGN.md",
        "docs/platform_designs/ensemble_analytics_db_v3_blueprint.md",
        "docs/rade_analytics/page_contract.md",
        "docs/rade_analytics/inference_implementation.md",
        "docs/rade_analytics/ensemble_infer_refactor.md",
        "docs/ensemble_implementation.md",
        "docs/ensemble_dashboard_design.md",
        "docs/hybrid_gnn_rnn_pipeline_suggested_changes.md",
        "docs/AI_IMPLEMENTATION.md",
        "src/rade_sr/PIPELINE.md",
        "src/rade_ml_pt",          # walk recursively for READMEs
        "src/rade_sr/README.md",
    ],
    "framework_guide": [
        "docs/guides",
        "docs/architecture",
    ],
    "user_doc": [
        "docs/guides/ui/dash_ui_tutorial.md",
        "docs/BEST_PRACTICES.md",
        "src/ui",                  # walk for READMEs
        "README.md",
        "UPDATE.md",
    ],
}

_EXCLUDES = ["*.aux", "*.out", "**/testing/**", "**/development/roadmap.md"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=list(_TIER_ROOTS), help="Index only one tier")
    parser.add_argument("--path", help="Index only this single path")
    parser.add_argument("--full", action="store_true", help="Wipe and rebuild from scratch")
    args = parser.parse_args()

    settings = get_settings()
    indexer = DocIndexer(
        store_path=settings.ai_rag_store_path,
        embed_model=settings.ai_rag_embed_model,
        chunk_size=settings.ai_rag_chunk_size,
        chunk_overlap=settings.ai_rag_chunk_overlap,
        ingestion_log_path=".cache/rade_ai/ingestion_log.jsonl",
    )

    if args.full:
        logger.info("Full rebuild requested — wiping existing index")
        indexer.client.delete_collection("rade_docs")
        indexer.collection = indexer.client.get_or_create_collection("rade_docs")

    t0 = time.time()
    total = 0
    if args.path:
        tier = args.tier or _infer_tier(args.path)
        total = indexer.ingest_file(args.path, tier)
    else:
        tiers = [args.tier] if args.tier else list(_TIER_ROOTS)
        for tier in tiers:
            for root in _TIER_ROOTS[tier]:
                if not Path(root).exists():
                    continue
                n = indexer.ingest_directory(root, tier, exclude_globs=_EXCLUDES)
                logger.info("tier=%s root=%s → %d chunks", tier, root, n)
                total += n

    stats = indexer.stats()
    logger.info("Done: %d new chunks · total %d · %.1fs",
                total, stats["n_chunks"], time.time() - t0)


def _infer_tier(path: str) -> str:
    for tier, roots in _TIER_ROOTS.items():
        for r in roots:
            if path.startswith(r):
                return tier
    return "user_doc"


if __name__ == "__main__":
    main()
```

## E.4 — Test smoke check

After running `python scripts/reindex_docs.py --full`, verify with:

```python
from rade_ml_pt.ensemble.api.services.ai_doc_retriever import DocRetriever

retriever = DocRetriever(store_path=".cache/rade_ai/chroma")
result = retriever.search("How does the residual MLP standardise outputs?", top_k=3)

for c in result["chunks"]:
    print(f"{c['similarity']:.3f}  {c['doc_path']} § {c['section_chain']}")
```

Expected: top hit should be in a `hybrid_gnn_rnn` reference doc, section about the residual MLP / standardisation. If you get unrelated chunks, the chunking is over-fragmenting and `chunk_size` should be increased.

---

## Closing notes

- **Build small, ship continuously.** Don't try to launch Tier 5 features on day one. Each phase makes the AI noticeably more useful — release each one to a small user group and gather feedback before moving on.
- **Audit log everything.** It's much cheaper to have it from day one than retrofit it after compliance asks.
- **Treat the LLM as a junior analyst.** Useful, fast, but needs supervision. Build the UI assuming a human will check every important output.
- **Iterate on prompts more than code.** 80% of perceived quality comes from system prompts. Set up a prompt testing harness early.
- **Talk to compliance EARLY.** Whether OpenAI can be called from a service is a policy question, not an engineering question. Get a written yes before you build.

Maintainer: `rade_analytics` team.

