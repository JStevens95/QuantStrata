<!--
  QNet deck - 04 Why you can trust it.

  Message: QNet makes the two things a validator asks about structural:
  a test score is honest because nothing is fitted on future data, and
  every result can be rebuilt from what was saved.

  A shuffled split shows the leak first (test days between training days);
  then QNet's forward split draws and the fitting bracket lands on the
  training window only. The bundle's files tick in one by one. The HTML
  has no blank lines on purpose: Markdown ends an HTML block at the first
  blank line.
-->

<style>
  .qs { container-type: inline-size; color: #e8eef5; font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
  .qs * { box-sizing: border-box; }
  .qs__frame {
    position: relative;
    aspect-ratio: 16 / 9;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #2a3340;
    background:
      radial-gradient(60% 80% at 22% 50%, rgba(77, 163, 255, 0.18), transparent 70%),
      radial-gradient(40% 60% at 85% 85%, rgba(63, 185, 80, 0.08), transparent 70%),
      #0d1117;
    padding: 4.2cqw 6cqw 5.4cqw;
    display: flex;
    flex-direction: column;
  }
  .qs__brand { position: absolute; top: 2.6cqw; right: 4cqw; width: 7cqw; height: auto; }
  .qs__kicker { margin: 0; font-size: 1cqw; font-weight: 700; letter-spacing: 0.22em; text-transform: uppercase; color: #4da3ff; }
  .qs__title { margin: 0.8cqw 0 0; max-width: 66%; font-size: 3.3cqw; line-height: 1.1; font-weight: 700; letter-spacing: -0.02em; }
  .qs__sub { margin: 1cqw 0 0; max-width: 60%; font-size: 1.55cqw; line-height: 1.35; color: #8ea3b8; }
  .qs__stage { position: relative; flex: 1; min-height: 0; margin-top: 2.4cqw; }
  .qs__foot {
    position: absolute; left: 6cqw; right: 6cqw; bottom: 2.4cqw;
    display: flex; justify-content: space-between;
    font-size: 0.95cqw; letter-spacing: 0.08em; color: #5d7086;
  }
  .qs-in { animation: qs-rise 0.6s cubic-bezier(0.2, 0.7, 0.2, 1) both; animation-delay: calc(var(--i, 0) * 90ms + 150ms); }
  @keyframes qs-rise { from { opacity: 0; transform: translateY(1.2cqw); } to { opacity: 1; transform: none; } }
  @media (prefers-reduced-motion: reduce) { .qs *, .qs *::after { animation: none !important; } }

  .qs04__stage { display: grid; grid-template-columns: 1fr 1fr; gap: 2cqw; }
  .qs04__panel {
    display: flex; flex-direction: column; padding: 2cqw 2.2cqw;
    border: 1px solid #2a3340; border-radius: 1.2cqw; background: rgba(22, 27, 34, 0.85);
  }
  .qs04__panel h2 { margin: 0; font-size: 1.75cqw; font-weight: 700; }
  .qs04__panel p { margin: 0.4cqw 0 0; font-size: 1.15cqw; color: #8ea3b8; }
  .qs04__panel svg { width: 100%; flex: 1; min-height: 0; margin-top: 1cqw; }
  .qs04__panel text { font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
  .qs04__bar { transform-box: fill-box; transform-origin: left center; animation: qs04-grow 0.5s ease-out both; animation-delay: var(--d); }
  @keyframes qs04-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
  .qs04__bracket { stroke-dasharray: 400; stroke-dashoffset: 400; animation: qs04-draw 0.8s ease-out 2.1s forwards; }
  @keyframes qs04-draw { to { stroke-dashoffset: 0; } }
  .qs04__late { animation: qs-rise 0.6s cubic-bezier(0.2, 0.7, 0.2, 1) both; animation-delay: var(--d); }
  .qs04__files { margin: 1.4cqw 0 0; padding: 0; list-style: none; display: grid; gap: 0.75cqw; }
  .qs04__files li {
    display: grid; grid-template-columns: 2.2cqw 13cqw 1fr; align-items: center;
    padding: 0.7cqw 1cqw; border: 1px solid #222b36; border-radius: 0.7cqw; background: #12171e;
    font-size: 1.1cqw; color: #8ea3b8;
    animation: qs-rise 0.5s cubic-bezier(0.2, 0.7, 0.2, 1) both; animation-delay: calc(var(--i) * 220ms + 600ms);
  }
  .qs04__files code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.1cqw; color: #e8eef5; }
  .qs04__files i {
    width: 1.4cqw; height: 1.4cqw; border-radius: 50%; background: #173d24; border: 1px solid #3fb950;
    display: grid; place-items: center; font-style: normal; font-size: 0.9cqw; color: #3fb950;
  }
</style>

<div class="qs qs04">
  <div class="qs__frame">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs04-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs04-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs04-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs04-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs04-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs04-brand-body)"/>
    <g clip-path="url(#qs04-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs04-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs04-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
    <path d="M16 124 L68 64"/>
    <path d="M16 124 L68 104"/>
    <path d="M16 124 L68 144"/>
    <path d="M16 124 L68 184"/>
    <path d="M68 64 L128 64"/>
    <path d="M68 64 L128 104"/>
    <path d="M68 64 L128 144"/>
    <path d="M68 64 L128 184"/>
    <path d="M68 104 L128 64"/>
    <path d="M68 104 L128 104"/>
    <path d="M68 104 L128 144"/>
    <path d="M68 104 L128 184"/>
    <path d="M68 144 L128 64"/>
    <path d="M68 144 L128 104"/>
    <path d="M68 144 L128 144"/>
    <path d="M68 144 L128 184"/>
    <path d="M68 184 L128 64"/>
    <path d="M68 184 L128 104"/>
    <path d="M68 184 L128 144"/>
    <path d="M68 184 L128 184"/>
    <path d="M128 64 L188 64"/>
    <path d="M128 64 L188 104"/>
    <path d="M128 64 L188 144"/>
    <path d="M128 64 L188 184"/>
    <path d="M128 104 L188 64"/>
    <path d="M128 104 L188 104"/>
    <path d="M128 104 L188 144"/>
    <path d="M128 104 L188 184"/>
    <path d="M128 144 L188 64"/>
    <path d="M128 144 L188 104"/>
    <path d="M128 144 L188 144"/>
    <path d="M128 144 L188 184"/>
    <path d="M128 184 L188 64"/>
    <path d="M128 184 L188 104"/>
    <path d="M128 184 L188 144"/>
    <path d="M128 184 L188 184"/>
    <path d="M188 64 L240 124"/>
    <path d="M188 104 L240 124"/>
    <path d="M188 144 L240 124"/>
    <path d="M188 184 L240 124"/>
    </g>
    <g stroke-linecap="round" stroke-linejoin="round">
    <path d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="#bfe0ff" stroke-width="6" stroke-opacity="0.22"/>
    <path id="qs04-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs04-brand-signal)" stroke-width="2.4"/>
    </g>
    <g fill="#4da3ff" stroke="#ffffff" stroke-width="2">
    <circle cx="68" cy="64" r="4.6"/>
    <circle cx="68" cy="104" r="4.6"/>
    <circle cx="68" cy="184" r="4.6"/>
    <circle cx="128" cy="64" r="4.6"/>
    <circle cx="128" cy="184" r="4.6"/>
    <circle cx="188" cy="64" r="4.6"/>
    <circle cx="188" cy="144" r="4.6"/>
    <circle cx="188" cy="184" r="4.6"/>
    </g>
    <g fill="#ffffff" stroke="#4da3ff" stroke-width="2">
    <circle cx="68" cy="144" r="9" fill-opacity="0.18" stroke="none"/>
    <circle cx="68" cy="144" r="5"/>
    <circle cx="188" cy="104" r="9" fill-opacity="0.18" stroke="none"/>
    <circle cx="188" cy="104" r="5"/>
    </g>
    <circle cx="16" cy="124" r="12" stroke="#4da3ff" stroke-width="1.5" stroke-opacity="0.45"/>
    <circle cx="16" cy="124" r="7" fill="#4da3ff" stroke="#ffffff" stroke-width="2.25"/>
    <circle cx="240" cy="124" r="13" stroke="#3fb950" stroke-width="1.5" stroke-opacity="0.5"/>
    <circle cx="240" cy="124" r="7.5" fill="#3fb950" stroke="#ffffff" stroke-width="2.25"/>
    <circle cx="128" cy="124" r="37" fill="url(#qs04-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">04 · Why you can trust it</p>
    <h1 class="qs__title qs-in" style="--i:1">Numbers a validator can sign off</h1>
    <p class="qs__sub qs-in" style="--i:2">Honest by construction. Rebuildable on demand.</p>
    <div class="qs__stage qs04__stage">
      <section class="qs04__panel qs-in" style="--i:3">
        <h2>Backtests you can believe</h2>
        <p>Training never sees the future, so a test score means what it says.</p>
        <svg viewBox="0 0 520 262" fill="none" role="img" aria-label="A shuffled split leaks future days into training and flatters the score; QNet splits forward in time and fits scalers on the training window only">
          <g class="qs04__late" style="--d:0.4s">
            <text x="20" y="18" fill="#f0a8a3" font-size="13" font-weight="700" letter-spacing="2">SHUFFLED SPLIT</text>
            <g fill-opacity="0.5">
              <rect x="20" y="30" width="44" height="26" rx="5" fill="#2f86e6"/>
              <rect x="68" y="30" width="20" height="26" rx="5" fill="#2e8a43"/>
              <rect x="92" y="30" width="64" height="26" rx="5" fill="#2f86e6"/>
              <rect x="160" y="30" width="20" height="26" rx="5" fill="#2e8a43"/>
              <rect x="184" y="30" width="80" height="26" rx="5" fill="#2f86e6"/>
              <rect x="268" y="30" width="20" height="26" rx="5" fill="#2e8a43"/>
              <rect x="292" y="30" width="52" height="26" rx="5" fill="#2f86e6"/>
              <rect x="348" y="30" width="20" height="26" rx="5" fill="#2e8a43"/>
              <rect x="372" y="30" width="60" height="26" rx="5" fill="#2f86e6"/>
            </g>
            <circle cx="458" cy="43" r="13" fill="#3a1a1c" stroke="#f85149" stroke-width="2"/>
            <path d="M452 37 L464 49 M464 37 L452 49" stroke="#f85149" stroke-width="2.4" stroke-linecap="round"/>
            <text x="20" y="80" fill="#8ea3b8" font-size="14">Test days sit between training days: the score is flattered.</text>
          </g>
          <path class="qs04__late" style="--d:0.8s" d="M20 100 H500" stroke="#2a3340" stroke-width="1"/>
          <text class="qs04__late" style="--d:0.9s" x="20" y="122" fill="#4da3ff" font-size="13" font-weight="700" letter-spacing="2">QNET: FORWARD IN TIME</text>
          <path class="qs04__bracket" d="M24 172 V160 H326 V172" stroke="#4da3ff" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
          <text class="qs04__late" style="--d:2.6s" x="175" y="152" fill="#cfe5ff" font-size="14" font-weight="600" text-anchor="middle">Scalers and feature selection fitted here only</text>
          <rect class="qs04__bar" style="--d:1.1s" x="20" y="180" width="310" height="40" rx="8" fill="#2f86e6"/>
          <rect class="qs04__bar" style="--d:1.5s" x="336" y="180" width="84" height="40" rx="8" fill="#4b5f78"/>
          <rect class="qs04__bar" style="--d:1.8s" x="426" y="180" width="74" height="40" rx="8" fill="#2e8a43"/>
          <g font-size="16" font-weight="600" fill="#e8eef5">
            <text class="qs04__late" style="--d:1.2s" x="175" y="206" text-anchor="middle">Train</text>
            <text class="qs04__late" style="--d:1.6s" x="378" y="206" text-anchor="middle">Validate</text>
            <text class="qs04__late" style="--d:1.9s" x="463" y="206" text-anchor="middle">Test</text>
          </g>
          <path d="M20 240 H492" stroke="#5d7086" stroke-width="1.5"/>
          <path d="M484 234 L494 240 L484 246" stroke="#5d7086" stroke-width="1.5" stroke-linejoin="round"/>
          <text x="20" y="260" fill="#5d7086" font-size="12" letter-spacing="2">TIME</text>
        </svg>
      </section>
      <section class="qs04__panel qs-in" style="--i:4">
        <h2>Every result rebuilds</h2>
        <p>A saved model carries everything needed to reproduce it.</p>
        <ul class="qs04__files">
          <li style="--i:1"><i>✓</i><code>weights</code>the trained model</li>
          <li style="--i:2"><i>✓</i><code>spec</code>the exact configuration</li>
          <li style="--i:3"><i>✓</i><code>fitted_state</code>scalers, encoders, graph</li>
          <li style="--i:4"><i>✓</i><code>lineage</code>data fingerprint, code version</li>
          <li style="--i:5"><i>✓</i><code>manifest</code>a hash of every file</li>
        </ul>
      </section>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>04</span></div>
  </div>
</div>
