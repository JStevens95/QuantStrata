# Q-Learning & RL Framework (Phase 7.2)

Technical reference for the QuantStrata Q-Learning / RL integration: protocols, pipelines, environments, and inference.

**Status:** Phase 7.2 core complete (protocols, pipelines, BaseEnv, metrics). See `docs/development/progress/phase_7_2_q_learning.md`.

---

## 1. Overview

The RL framework is **agent- and environment-agnostic**:

1. **Build agent** — Any object implementing `RLAgent` (select_action, update, get/set_parameters).
2. **Build or wrap environment** — Any object implementing `RLEnvironment` (reset, step).
3. **Train** — `run_training(agent, env, config)` runs the generic episode loop.
4. **Evaluate** — `evaluate_agent(agent, env, n_episodes)` returns `RLEvaluationResult` (mean return, Sharpe, drawdown, win rate).
5. **Deploy** — `save_agent()` / `load_agent()` and `select_action(agent, state)` for backtest or live.

---

## 2. Module Layout

| Path | Purpose |
|------|---------|
| `core/protocols.py` | `RLEnvironment`, `RLAgent` |
| `core/types.py` | `Transition`, `RLTrainingConfig`, `RLTrainingResult`, `RLEvaluationResult` |
| `pipelines/training.py` | `run_training()`, `RLTrainingLoop` |
| `pipelines/evaluation.py` | `evaluate_agent()` |
| `pipelines/inference.py` | `save_agent()`, `load_agent()`, `select_action()` |
| `environments/base.py` | `BaseEnv` (minimal env for tests / template) |
| `evaluation/metrics.py` | `sharpe_ratio`, `max_drawdown`, `win_rate` |

---

## 3. Protocols

### RLEnvironment

- `reset(*, seed=None, options=None) -> (state, info)`
- `step(action) -> (state, reward, terminated, truncated, info)`

State/action can be any type (e.g. ndarray, dict). `terminated` = natural end (goal/failure), `truncated` = time limit or cut.

### RLAgent

- `select_action(state, *, training=False, explore=True) -> action`
- `update(transitions=None, batch=None) -> Optional[dict]` (metrics)
- `get_parameters() -> dict`
- `set_parameters(params: dict) -> None`

---

## 4. Types

- **Transition:** state, action, reward, next_state, terminated, truncated, info.
- **RLTrainingConfig:** n_episodes, max_steps_per_episode, learning_rate, gamma, checkpoint_dir, checkpoint_frequency, save_best_only, log_every, eval_episodes, verbose.
- **RLTrainingResult:** episode_returns, episode_lengths, history, best_episode_return, best_episode, config, training_time_seconds, metadata; `to_json()` / `from_json()`.
- **RLEvaluationResult:** mean_return, std_return, mean_length, returns, lengths, metrics (e.g. sharpe, max_drawdown, win_rate).

---

## 5. Training

- **Entry point:** `run_training(agent, env, config)`.
- **Loop:** For each episode: env.reset() → loop step with agent.select_action(..., explore=True) → collect list of `Transition` → agent.update(transitions=...). Optional eval episodes (explore=False) every log_every. Checkpointing: best only or periodic to JSON (agent parameters).

---

## 6. Evaluation

- **Entry point:** `evaluate_agent(agent, env, n_episodes=10, max_steps_per_episode=0, explore=False, metrics=None)`.
- **Default metrics:** sharpe_ratio, max_drawdown, win_rate (over episode returns).
- **Output:** `RLEvaluationResult` with mean/std return, mean length, per-episode returns/lengths, and metrics dict.

---

## 7. Inference

- **Save:** `save_agent(agent, artifact_dir, config=None, metadata=None)` writes parameters.json, config.json, metadata.json.
- **Load:** `load_agent(artifact_dir, agent_factory, factory_kwargs=None)` loads parameters and calls agent.set_parameters(); agent_factory is e.g. a class or constructor.
- **Deploy:** `select_action(agent, state, training=False, explore=False)` for single-step action selection in backtest or live.

---

## 8. References

- Progress: `docs/development/progress/phase_7_2_q_learning.md`
- Roadmap: `docs/development/roadmap.md` (Phase 7.2)
- Guide: [RL Framework Guide](../../guides/q_learning/rl_framework.md)
