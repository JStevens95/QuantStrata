<!--
  QNet deck - 03 What you can build.

  Message: QNet is not one model. Twelve examples of the kind of model the
  desks already need, across front office, trading, market risk, XVA and
  counterparty risk, each onboarded the same way.

  Three pages of four cards. The buttons and dots at the top right move
  between pages; they are hidden radio inputs and their labels, so they
  work with no script. Each card loops its own small diagram. The HTML has
  no blank lines on purpose: Markdown ends an HTML block at the first
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
  @media (prefers-reduced-motion: reduce) { .qs *, .qs *::after { animation: none !important; transition: none !important; } }

  .qs03__page-input { position: absolute; opacity: 0; pointer-events: none; }
  .qs03__nav { position: absolute; top: 11.6cqw; right: 6cqw; z-index: 2; display: flex; align-items: center; gap: 0.7cqw; }
  .qs03__go {
    display: none; align-items: center;
    padding: 0.5cqw 1.3cqw; border: 1px solid #2d5680; border-radius: 999px;
    background: rgba(77, 163, 255, 0.1); color: #cfe5ff;
    font-size: 1.05cqw; font-weight: 600; cursor: pointer; user-select: none;
  }
  .qs03__go:hover { background: rgba(77, 163, 255, 0.2); }
  .qs03__dots { display: flex; gap: 0.4cqw; margin-left: 0.3cqw; }
  .qs03__dots label { width: 0.6cqw; height: 0.6cqw; border-radius: 50%; background: #2d3b4d; cursor: pointer; }
  #qs03-p1:checked ~ .qs03__nav .qs03__go--p1,
  #qs03-p2:checked ~ .qs03__nav .qs03__go--p2,
  #qs03-p3:checked ~ .qs03__nav .qs03__go--p3 { display: inline-flex; }
  #qs03-p1:checked ~ .qs03__nav .qs03__dots label:nth-child(1),
  #qs03-p2:checked ~ .qs03__nav .qs03__dots label:nth-child(2),
  #qs03-p3:checked ~ .qs03__nav .qs03__dots label:nth-child(3) { background: #4da3ff; }
  .qs03__viewport { overflow: hidden; }
  .qs03__track { display: flex; width: 300%; height: 100%; transition: transform 0.75s cubic-bezier(0.65, 0, 0.25, 1); }
  #qs03-p2:checked ~ .qs03__viewport .qs03__track { transform: translateX(-33.3333%); }
  #qs03-p3:checked ~ .qs03__viewport .qs03__track { transform: translateX(-66.6667%); }
  .qs03__page { width: 33.3333%; height: 100%; display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; gap: 1.4cqw; }
  .qs03__card {
    display: grid; grid-template-columns: 44% 1fr; align-items: center; gap: 1.6cqw;
    padding: 1.2cqw 1.6cqw; border: 1px solid #2a3340; border-radius: 1.2cqw;
    background: rgba(22, 27, 34, 0.85); border-left: 0.3cqw solid var(--c);
  }
  .qs03__card svg { width: 100%; height: auto; }
  .qs03__card text { font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
  .qs03__tag { margin: 0; font-size: 0.85cqw; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--c); }
  .qs03__card h2 { margin: 0.35cqw 0 0; font-size: 1.55cqw; line-height: 1.2; font-weight: 700; }
  .qs03__card p.qs03__what { margin: 0.35cqw 0 0; font-size: 1.05cqw; line-height: 1.35; color: #8ea3b8; }
  .qs03__model {
    display: inline-flex; align-items: center; gap: 0.5cqw; margin-top: 0.7cqw;
    padding: 0.25cqw 0.8cqw; border: 1px solid #2d5680; border-radius: 999px;
    background: rgba(77, 163, 255, 0.08); font-size: 0.95cqw; font-weight: 600; color: #cfe5ff;
  }
  .qs03__draw { stroke-dasharray: var(--len); stroke-dashoffset: var(--len); animation: qs03-draw var(--t, 4s) ease-in-out infinite; animation-delay: var(--d, 0s); }
  @keyframes qs03-draw { 0% { stroke-dashoffset: var(--len); } 45%, 85% { stroke-dashoffset: 0; } 100% { stroke-dashoffset: 0; opacity: 0; } }
  .qs03__pop { transform-box: fill-box; transform-origin: center; animation: qs03-pop var(--t, 4s) ease-out infinite; animation-delay: var(--d); }
  @keyframes qs03-pop { 0%, 20% { transform: scale(0); opacity: 0; } 30%, 85% { transform: scale(1); opacity: 1; } 100% { opacity: 0; } }
  .qs03__lit { animation: qs03-lit 3s ease-in-out infinite; animation-delay: var(--d); }
  @keyframes qs03-lit { 0%, 100% { fill: #2a3340; } 30%, 60% { fill: #d29922; } }
  .qs03__fade { animation: qs03-fade var(--t, 4s) ease-in-out infinite; animation-delay: var(--d, 0s); }
  @keyframes qs03-fade { 0%, 35% { opacity: 0; } 50%, 85% { opacity: 1; } 100% { opacity: 0; } }
  .qs03__bar { transform-box: fill-box; transform-origin: bottom; animation: qs03-bar 4s ease-out infinite; animation-delay: var(--d); }
  @keyframes qs03-bar { 0%, 10% { transform: scaleY(0); } 25%, 85% { transform: scaleY(1); } 100% { transform: scaleY(1); opacity: 0; } }
</style>

<div class="qs qs03">
  <div class="qs__frame">
    <input type="radio" name="qs03-page" id="qs03-p1" class="qs03__page-input" checked>
    <input type="radio" name="qs03-page" id="qs03-p2" class="qs03__page-input">
    <input type="radio" name="qs03-page" id="qs03-p3" class="qs03__page-input">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs03-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs03-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs03-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs03-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs03-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs03-brand-body)"/>
    <g clip-path="url(#qs03-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs03-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs03-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
    <path id="qs03-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs03-brand-signal)" stroke-width="2.4"/>
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
    <circle cx="128" cy="124" r="37" fill="url(#qs03-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">03 · What you can build</p>
    <h1 class="qs__title qs-in" style="--i:1">One platform. The models your desks already need.</h1>
    <div class="qs03__nav qs-in" style="--i:2">
      <label for="qs03-p1" class="qs03__go qs03__go--p2">← Back</label>
      <label for="qs03-p2" class="qs03__go qs03__go--p3">← Back</label>
      <label for="qs03-p2" class="qs03__go qs03__go--p1">4 more ideas →</label>
      <label for="qs03-p3" class="qs03__go qs03__go--p2">4 more →</label>
      <span class="qs03__dots"><label for="qs03-p1" aria-label="Page 1"></label><label for="qs03-p2" aria-label="Page 2"></label><label for="qs03-p3" aria-label="Page 3"></label></span>
    </div>
    <div class="qs__stage qs03__viewport">
      <div class="qs03__track">
        <div class="qs03__page">
          <section class="qs03__card qs-in" style="--i:2; --c:#4da3ff">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="A yield curve fitted through quoted tenors, with missing tenors backfilled">
            <path d="M16 18 V134 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <path class="qs03__draw" style="--len:247" d="M22.0 119.5 L26.0 114.8 L30.0 110.3 L34.0 106.0 L38.0 102.0 L42.0 98.3 L46.0 94.7 L50.0 91.3 L54.0 88.1 L58.0 85.1 L62.0 82.3 L66.0 79.6 L70.0 77.1 L74.0 74.7 L78.0 72.4 L82.0 70.3 L86.0 68.3 L90.0 66.4 L94.0 64.6 L98.0 62.9 L102.0 61.3 L106.0 59.8 L110.0 58.3 L114.0 57.0 L118.0 55.7 L122.0 54.5 L126.0 53.4 L130.0 52.3 L134.0 51.3 L138.0 50.3 L142.0 49.4 L146.0 48.5 L150.0 47.7 L154.0 47.0 L158.0 46.3 L162.0 45.6 L166.0 44.9 L170.0 44.3 L174.0 43.8 L178.0 43.2 L182.0 42.7 L186.0 42.2 L190.0 41.8 L194.0 41.3 L198.0 40.9 L202.0 40.5 L206.0 40.2 L210.0 39.8 L214.0 39.5 L218.0 39.2 L222.0 38.9 L226.0 38.6 L230.0 38.4 L234.0 38.1 L238.0 37.9 L242.0 37.7 L246.0 37.5" stroke="#4da3ff" stroke-width="2.5" stroke-linecap="round"/>
            <circle cx="28" cy="112.5" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="52" cy="89.7" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="98" cy="62.9" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="140" cy="49.8" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="196" cy="41.1" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="240" cy="37.8" r="5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.00s" cx="75" cy="74.1" r="5" fill="#0d1117" stroke="#3fb950" stroke-width="2.5"/>
            <circle class="qs03__pop" style="--d:0.15s" cx="120" cy="55.1" r="5" fill="#0d1117" stroke="#3fb950" stroke-width="2.5"/>
            <circle class="qs03__pop" style="--d:0.30s" cx="168" cy="44.6" r="5" fill="#0d1117" stroke="#3fb950" stroke-width="2.5"/>
            <circle class="qs03__pop" style="--d:0.45s" cx="218" cy="39.2" r="5" fill="#0d1117" stroke="#3fb950" stroke-width="2.5"/>
            <text x="250" y="148" fill="#5d7086" font-size="11" text-anchor="end" letter-spacing="1.5">TENOR</text>
            <circle cx="24" cy="146" r="3.5" fill="#4da3ff"/><text x="31" y="149" fill="#8ea3b8" font-size="10">quoted</text>
            <circle cx="80" cy="146" r="3.5" fill="none" stroke="#3fb950" stroke-width="2"/><text x="87" y="149" fill="#8ea3b8" font-size="10">backfilled</text>
            </svg>
            <div>
              <p class="qs03__tag">Front office · Trading</p>
              <h2>Curve bootstrap and backfill</h2>
              <p class="qs03__what">Fill missing tenors and history from the points you trust.</p>
              <span class="qs03__model">Ridge regression · scikit-learn</span>
            </div>
          </section>
          <section class="qs03__card qs-in" style="--i:3; --c:#3fb950">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="A graph of instruments feeds a proxy that traces the full revaluation P&amp;L">
            <g stroke="#ffffff" stroke-opacity="0.35" stroke-width="1.5">
            <path d="M20 72 L48 50"/>
            <path d="M20 72 L48 96"/>
            <path d="M48 50 L76 72"/>
            <path d="M48 96 L76 72"/>
            <path d="M48 50 L48 96"/>
            </g>
            <circle cx="20" cy="72" r="5.5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="48" cy="50" r="5.5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="48" cy="96" r="5.5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="76" cy="72" r="5.5" fill="#4da3ff" stroke="#ffffff" stroke-width="1.5"/>
            <path d="M88 72 H108" stroke="#5d7086" stroke-width="1.5"/><path d="M103 67 L109 72 L103 77" stroke="#5d7086" stroke-width="1.5" stroke-linejoin="round"/>
            <path d="M120.0 75.0 L123.0 67.0 L126.0 59.9 L129.0 54.3 L132.0 50.8 L135.0 49.6 L138.0 50.5 L141.0 53.3 L144.0 57.2 L147.0 61.6 L150.0 65.8 L153.0 69.1 L156.0 71.3 L159.0 72.1 L162.0 71.7 L165.0 70.6 L168.0 69.3 L171.0 68.5 L174.0 68.9 L177.0 70.8 L180.0 74.5 L183.0 80.0 L186.0 86.9 L189.0 94.6 L192.0 102.4 L195.0 109.3 L198.0 114.7 L201.0 118.0 L204.0 118.8 L207.0 117.2 L210.0 113.3 L213.0 107.9 L216.0 101.6 L219.0 95.3 L222.0 89.7 L225.0 85.3 L228.0 82.5 L231.0 81.4 L234.0 81.7 L237.0 83.0 L240.0 84.6 L243.0 85.9 L246.0 86.3 L249.0 85.4" stroke="#8ea3b8" stroke-opacity="0.7" stroke-width="1.5" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:202" d="M120.0 72.7 L123.0 65.8 L126.0 60.1 L129.0 55.9 L132.0 53.2 L135.0 52.0 L138.0 52.1 L141.0 53.5 L144.0 55.9 L147.0 59.3 L150.0 63.3 L153.0 67.3 L156.0 70.7 L159.0 73.0 L162.0 73.8 L165.0 73.1 L168.0 71.3 L171.0 69.4 L174.0 68.3 L177.0 68.9 L180.0 72.0 L183.0 77.8 L186.0 85.7 L189.0 94.9 L192.0 104.0 L195.0 111.7 L198.0 117.1 L201.0 119.5 L204.0 118.9 L207.0 115.8 L210.0 111.0 L213.0 105.5 L216.0 99.9 L219.0 94.8 L222.0 90.7 L225.0 87.4 L228.0 85.0 L231.0 83.4 L234.0 82.5 L237.0 82.3 L240.0 82.7 L243.0 83.4 L246.0 84.1 L249.0 84.3" stroke="#3fb950" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M20 146 H34" stroke="#8ea3b8" stroke-width="1.5"/><text x="39" y="149" fill="#8ea3b8" font-size="10">full revaluation</text>
            <path d="M130 146 H144" stroke="#3fb950" stroke-width="2.5"/><text x="149" y="149" fill="#8ea3b8" font-size="10">proxy</text>
            </svg>
            <div>
              <p class="qs03__tag">Market risk · CCR</p>
              <h2>Full-revaluation proxy</h2>
              <p class="qs03__what">Learn the book's P&amp;L so scenarios run without a full reval.</p>
              <span class="qs03__model">Hybrid GNN-RNN · PyTorch</span>
            </div>
          </section>
          <section class="qs03__card qs-in" style="--i:4; --c:#d29922">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Three stressed drivers expand to a full set of risk factors">
            <g stroke="#d29922" stroke-opacity="0.18" stroke-width="1.2">
            <path d="M40 36 L192 18"/>
            <path d="M40 36 L228 18"/>
            <path d="M40 36 L192 38"/>
            <path d="M40 36 L228 38"/>
            <path d="M40 36 L192 58"/>
            <path d="M40 36 L228 58"/>
            <path d="M40 36 L192 78"/>
            <path d="M40 36 L228 78"/>
            <path d="M40 36 L192 98"/>
            <path d="M40 36 L228 98"/>
            <path d="M40 36 L192 118"/>
            <path d="M40 36 L228 118"/>
            <path d="M40 72 L192 18"/>
            <path d="M40 72 L228 18"/>
            <path d="M40 72 L192 38"/>
            <path d="M40 72 L228 38"/>
            <path d="M40 72 L192 58"/>
            <path d="M40 72 L228 58"/>
            <path d="M40 72 L192 78"/>
            <path d="M40 72 L228 78"/>
            <path d="M40 72 L192 98"/>
            <path d="M40 72 L228 98"/>
            <path d="M40 72 L192 118"/>
            <path d="M40 72 L228 118"/>
            <path d="M40 108 L192 18"/>
            <path d="M40 108 L228 18"/>
            <path d="M40 108 L192 38"/>
            <path d="M40 108 L228 38"/>
            <path d="M40 108 L192 58"/>
            <path d="M40 108 L228 58"/>
            <path d="M40 108 L192 78"/>
            <path d="M40 108 L228 78"/>
            <path d="M40 108 L192 98"/>
            <path d="M40 108 L228 98"/>
            <path d="M40 108 L192 118"/>
            <path d="M40 108 L228 118"/>
            </g>
            <circle cx="40" cy="36" r="8" fill="#d29922" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="40" cy="72" r="8" fill="#d29922" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="40" cy="108" r="8" fill="#d29922" stroke="#ffffff" stroke-width="1.5"/>
            <circle class="qs03__lit" style="--d:0.00s" cx="192" cy="18" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.12s" cx="228" cy="18" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.24s" cx="192" cy="38" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.36s" cx="228" cy="38" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.48s" cx="192" cy="58" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.60s" cx="228" cy="58" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.72s" cx="192" cy="78" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.84s" cx="228" cy="78" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:0.96s" cx="192" cy="98" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:1.08s" cx="228" cy="98" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:1.20s" cx="192" cy="118" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <circle class="qs03__lit" style="--d:1.32s" cx="228" cy="118" r="5" fill="#2a3340" stroke="#5d7086" stroke-width="1"/>
            <text x="40" y="146" fill="#8ea3b8" font-size="10" text-anchor="middle">3 shocks</text>
            <text x="210" y="146" fill="#8ea3b8" font-size="10" text-anchor="middle">full factor set</text>
            </svg>
            <div>
              <p class="qs03__tag">Stress testing · XVA</p>
              <h2>Risk factor expansion</h2>
              <p class="qs03__what">Turn a handful of stressed drivers into the full factor set.</p>
              <span class="qs03__model">Gradient boosting · XGBoost</span>
            </div>
          </section>
          <section class="qs03__card qs-in" style="--i:5; --c:#a78bfa">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Hedged P&amp;L with transaction costs: the learned hedge has a tighter distribution and a smaller loss tail than the delta hedge">
            <path d="M16 124 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <rect class="qs03__bar" style="--d:0.00s" x="20.6" y="123.4" width="6.9" height="0.6" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.03s" x="28.7" y="123.0" width="6.9" height="1.0" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.05s" x="36.7" y="122.3" width="6.9" height="1.7" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.08s" x="44.8" y="121.5" width="6.9" height="2.5" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.10s" x="52.9" y="120.4" width="6.9" height="3.6" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.12s" x="61.0" y="119.2" width="6.9" height="4.8" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.15s" x="69.0" y="117.7" width="6.9" height="6.3" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.18s" x="77.1" y="115.9" width="6.9" height="8.1" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.20s" x="85.2" y="113.9" width="6.9" height="10.1" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.23s" x="93.2" y="111.3" width="6.9" height="12.7" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.25s" x="101.3" y="107.8" width="6.9" height="16.2" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.28s" x="109.4" y="103.2" width="6.9" height="20.8" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.30s" x="117.5" y="97.3" width="6.9" height="26.7" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.33s" x="125.5" y="90.7" width="6.9" height="33.3" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.35s" x="133.6" y="84.1" width="6.9" height="39.9" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.38s" x="141.7" y="78.8" width="6.9" height="45.2" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.40s" x="149.7" y="76.0" width="6.9" height="48.0" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.43s" x="157.8" y="76.6" width="6.9" height="47.4" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.45s" x="165.9" y="80.5" width="6.9" height="43.5" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.48s" x="174.0" y="87.1" width="6.9" height="36.9" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.50s" x="182.0" y="95.1" width="6.9" height="28.9" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.53s" x="190.1" y="103.1" width="6.9" height="20.9" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.55s" x="198.2" y="110.0" width="6.9" height="14.0" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.58s" x="206.2" y="115.4" width="6.9" height="8.6" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.60s" x="214.3" y="119.1" width="6.9" height="4.9" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.62s" x="222.4" y="121.4" width="6.9" height="2.6" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.65s" x="230.5" y="122.8" width="6.9" height="1.2" rx="1" fill="#8ea3b8" fill-opacity="0.5"/>
            <rect class="qs03__bar" style="--d:0.82s" x="94.4" y="123.3" width="4.5" height="0.7" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.85s" x="102.5" y="122.5" width="4.5" height="1.5" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.88s" x="110.6" y="121.2" width="4.5" height="2.8" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.90s" x="118.6" y="119.2" width="4.5" height="4.8" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.93s" x="126.7" y="115.4" width="4.5" height="8.6" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.95s" x="134.8" y="107.3" width="4.5" height="16.7" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:0.97s" x="142.8" y="91.6" width="4.5" height="32.4" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.00s" x="150.9" y="67.9" width="4.5" height="56.1" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.02s" x="159.0" y="43.6" width="4.5" height="80.4" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.05s" x="167.1" y="32.0" width="4.5" height="92.0" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.07s" x="175.1" y="40.9" width="4.5" height="83.1" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.10s" x="183.2" y="65.0" width="4.5" height="59.0" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.12s" x="191.3" y="91.2" width="4.5" height="32.8" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.15s" x="199.3" y="109.8" width="4.5" height="14.2" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.18s" x="207.4" y="119.2" width="4.5" height="4.8" rx="1" fill="#a78bfa"/>
            <rect class="qs03__bar" style="--d:1.20s" x="215.5" y="122.7" width="4.5" height="1.3" rx="1" fill="#a78bfa"/>
            <path class="qs03__fade" style="--d:0.2s" d="M62.3 22 V124" stroke="#8ea3b8" stroke-width="1.2" stroke-dasharray="3 3"/>
            <path class="qs03__fade" style="--d:0.4s" d="M124.2 22 V124" stroke="#a78bfa" stroke-width="1.2" stroke-dasharray="3 3"/>
            <text class="qs03__fade" style="--d:0.4s" x="93.2" y="18" fill="#cfd8e3" font-size="9" text-anchor="middle">tail loss cut</text>
            <path class="qs03__fade" style="--d:0.4s" d="M65.3 26 H121.2 M117.2 23 L121.2 26 L117.2 29" stroke="#cfd8e3" stroke-width="1.2"/>
            <text x="20" y="136" fill="#5d7086" font-size="9" letter-spacing="1.5">LOSS</text>
            <text x="250" y="136" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">GAIN</text>
            <text x="250" y="12" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">HEDGED P&amp;L</text>
            <rect x="20" y="142" width="10" height="7" rx="1" fill="#8ea3b8" fill-opacity="0.45"/><text x="34" y="149" fill="#8ea3b8" font-size="9">delta hedge</text>
            <rect x="94" y="142" width="10" height="7" rx="1" fill="#a78bfa"/><text x="108" y="149" fill="#8ea3b8" font-size="9">deep hedge</text>
            <path d="M168 146 H180" stroke="#8ea3b8" stroke-width="1.2" stroke-dasharray="3 3"/><text x="184" y="149" fill="#8ea3b8" font-size="9">CVaR 95%</text>
            </svg>
            <div>
              <p class="qs03__tag">Trading · FO</p>
              <h2>Deep hedging</h2>
              <p class="qs03__what">Hedge the whole book under costs and limits, beyond Black–Scholes delta.</p>
              <span class="qs03__model">Neural hedging policy · PyTorch</span>
            </div>
          </section>
        </div>
        <div class="qs03__page">
          <section class="qs03__card" style="--c:#22d3ee">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Simulated risk factor paths fan out from today, with a 5 to 95 percent band">
            <path d="M16 12 V134 H250" stroke="#2a3340" stroke-width="1"/>
            <path class="qs03__fade" style="--t:5s" d="M22.0 78.0 L27.6 69.7 L33.2 66.9 L38.8 63.8 L44.4 60.5 L50.0 60.8 L55.6 58.1 L61.2 58.6 L66.8 56.3 L72.4 55.1 L78.0 54.0 L83.6 53.9 L89.2 51.6 L94.8 50.6 L100.4 51.3 L106.0 48.8 L111.6 48.5 L117.2 47.3 L122.8 46.8 L128.4 48.1 L134.0 46.1 L139.6 45.2 L145.2 45.1 L150.8 43.1 L156.4 42.0 L162.0 43.4 L167.6 43.2 L173.2 43.2 L178.8 40.9 L184.4 41.1 L190.0 39.3 L195.6 38.0 L201.2 38.1 L206.8 40.0 L212.4 39.7 L218.0 39.5 L223.6 38.2 L229.2 37.4 L234.8 38.9 L240.4 37.3 L246.0 38.6 L246.0 106.6 L240.4 103.7 L234.8 105.8 L229.2 103.2 L223.6 104.8 L218.0 104.6 L212.4 105.5 L206.8 107.1 L201.2 106.1 L195.6 106.0 L190.0 105.0 L184.4 106.6 L178.8 106.9 L173.2 106.3 L167.6 106.2 L162.0 103.6 L156.4 103.4 L150.8 104.9 L145.2 103.7 L139.6 103.5 L134.0 103.2 L128.4 104.5 L122.8 102.8 L117.2 105.2 L111.6 105.2 L106.0 100.4 L100.4 99.4 L94.8 99.1 L89.2 98.9 L83.6 97.7 L78.0 97.9 L72.4 97.1 L66.8 94.5 L61.2 93.8 L55.6 92.8 L50.0 93.9 L44.4 91.5 L38.8 88.8 L33.2 87.7 L27.6 85.2 L22.0 78.0 Z" fill="#22d3ee" fill-opacity="0.12"/>
            <path class="qs03__draw" style="--len:263; --t:5s; --d:0.00s" d="M22.0 78.0 L27.6 78.9 L33.2 76.4 L38.8 77.2 L44.4 78.4 L50.0 82.3 L55.6 83.0 L61.2 77.7 L66.8 75.5 L72.4 70.6 L78.0 69.4 L83.6 67.5 L89.2 66.6 L94.8 74.1 L100.4 70.0 L106.0 67.6 L111.6 65.3 L117.2 72.9 L122.8 80.6 L128.4 84.4 L134.0 86.1 L139.6 84.4 L145.2 84.3 L150.8 81.6 L156.4 84.2 L162.0 82.5 L167.6 80.4 L173.2 83.1 L178.8 75.1 L184.4 72.3 L190.0 66.8 L195.6 69.5 L201.2 72.8 L206.8 74.2 L212.4 74.5 L218.0 71.4 L223.6 70.2 L229.2 72.1 L234.8 76.3 L240.4 78.4 L246.0 72.6" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:289; --t:5s; --d:0.08s" d="M22.0 78.0 L27.6 81.4 L33.2 80.0 L38.8 77.8 L44.4 84.3 L50.0 83.8 L55.6 77.6 L61.2 86.4 L66.8 87.5 L72.4 87.6 L78.0 91.0 L83.6 88.3 L89.2 88.2 L94.8 94.4 L100.4 90.2 L106.0 86.7 L111.6 82.1 L117.2 75.3 L122.8 73.5 L128.4 72.8 L134.0 78.5 L139.6 75.5 L145.2 78.0 L150.8 79.8 L156.4 85.3 L162.0 89.3 L167.6 91.3 L173.2 85.1 L178.8 93.9 L184.4 100.0 L190.0 98.4 L195.6 91.3 L201.2 88.3 L206.8 96.5 L212.4 107.3 L218.0 105.1 L223.6 107.7 L229.2 112.1 L234.8 107.0 L240.4 101.3 L246.0 100.0" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:282; --t:5s; --d:0.16s" d="M22.0 78.0 L27.6 76.7 L33.2 74.5 L38.8 67.1 L44.4 64.2 L50.0 61.8 L55.6 59.4 L61.2 66.5 L66.8 60.7 L72.4 56.4 L78.0 54.1 L83.6 63.1 L89.2 66.0 L94.8 62.1 L100.4 70.3 L106.0 71.0 L111.6 66.3 L117.2 72.1 L122.8 64.7 L128.4 62.2 L134.0 62.9 L139.6 61.4 L145.2 58.5 L150.8 58.0 L156.4 52.9 L162.0 56.0 L167.6 58.0 L173.2 53.3 L178.8 53.3 L184.4 57.5 L190.0 53.3 L195.6 46.8 L201.2 49.0 L206.8 55.5 L212.4 56.2 L218.0 56.9 L223.6 58.4 L229.2 52.1 L234.8 56.9 L240.4 51.3 L246.0 57.2" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:272; --t:5s; --d:0.24s" d="M22.0 78.0 L27.6 81.3 L33.2 78.2 L38.8 72.8 L44.4 68.8 L50.0 67.1 L55.6 66.4 L61.2 65.7 L66.8 63.0 L72.4 63.8 L78.0 62.5 L83.6 60.0 L89.2 60.0 L94.8 56.6 L100.4 54.1 L106.0 45.1 L111.6 43.9 L117.2 46.1 L122.8 48.1 L128.4 48.3 L134.0 44.4 L139.6 46.2 L145.2 44.7 L150.8 36.6 L156.4 48.6 L162.0 53.9 L167.6 52.9 L173.2 51.3 L178.8 50.3 L184.4 52.5 L190.0 49.7 L195.6 48.6 L201.2 51.1 L206.8 40.3 L212.4 39.0 L218.0 41.9 L223.6 42.7 L229.2 44.0 L234.8 44.5 L240.4 57.1 L246.0 59.4" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:293; --t:5s; --d:0.32s" d="M22.0 78.0 L27.6 73.2 L33.2 78.3 L38.8 78.4 L44.4 73.8 L50.0 69.8 L55.6 62.9 L61.2 70.6 L66.8 72.1 L72.4 73.5 L78.0 70.5 L83.6 65.5 L89.2 77.5 L94.8 72.4 L100.4 78.8 L106.0 75.5 L111.6 82.0 L117.2 80.9 L122.8 75.2 L128.4 75.7 L134.0 74.7 L139.6 70.9 L145.2 70.1 L150.8 70.4 L156.4 63.3 L162.0 58.6 L167.6 60.0 L173.2 47.6 L178.8 53.0 L184.4 49.0 L190.0 50.4 L195.6 50.0 L201.2 47.0 L206.8 46.2 L212.4 43.6 L218.0 50.7 L223.6 57.7 L229.2 55.0 L234.8 59.5 L240.4 64.2 L246.0 70.8" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:290; --t:5s; --d:0.40s" d="M22.0 78.0 L27.6 72.0 L33.2 68.5 L38.8 61.8 L44.4 66.0 L50.0 66.0 L55.6 71.1 L61.2 67.5 L66.8 60.2 L72.4 64.3 L78.0 57.2 L83.6 52.8 L89.2 53.7 L94.8 62.8 L100.4 56.4 L106.0 56.9 L111.6 59.7 L117.2 58.0 L122.8 56.2 L128.4 49.5 L134.0 54.3 L139.6 49.3 L145.2 42.8 L150.8 36.5 L156.4 37.7 L162.0 41.4 L167.6 37.1 L173.2 37.0 L178.8 36.8 L184.4 30.8 L190.0 32.4 L195.6 43.3 L201.2 45.3 L206.8 53.9 L212.4 50.4 L218.0 49.1 L223.6 52.1 L229.2 52.3 L234.8 48.7 L240.4 48.5 L246.0 42.7" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:284; --t:5s; --d:0.48s" d="M22.0 78.0 L27.6 78.0 L33.2 73.1 L38.8 66.2 L44.4 58.9 L50.0 62.0 L55.6 58.0 L61.2 66.5 L66.8 71.4 L72.4 80.1 L78.0 75.0 L83.6 80.4 L89.2 80.2 L94.8 80.8 L100.4 80.6 L106.0 83.0 L111.6 81.7 L117.2 73.3 L122.8 72.9 L128.4 70.4 L134.0 65.7 L139.6 66.6 L145.2 72.2 L150.8 74.6 L156.4 69.5 L162.0 76.9 L167.6 79.3 L173.2 74.5 L178.8 70.8 L184.4 70.6 L190.0 66.8 L195.6 66.0 L201.2 71.3 L206.8 78.2 L212.4 80.9 L218.0 76.4 L223.6 78.8 L229.2 82.6 L234.8 85.8 L240.4 92.4 L246.0 92.5" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:292; --t:5s; --d:0.56s" d="M22.0 78.0 L27.6 83.1 L33.2 81.1 L38.8 91.5 L44.4 89.6 L50.0 92.1 L55.6 100.4 L61.2 96.6 L66.8 97.3 L72.4 106.9 L78.0 110.2 L83.6 108.1 L89.2 109.5 L94.8 105.3 L100.4 101.3 L106.0 97.7 L111.6 95.7 L117.2 89.1 L122.8 85.8 L128.4 83.4 L134.0 92.5 L139.6 88.0 L145.2 81.7 L150.8 82.7 L156.4 84.5 L162.0 75.4 L167.6 83.2 L173.2 80.8 L178.8 69.5 L184.4 73.6 L190.0 70.3 L195.6 61.7 L201.2 62.2 L206.8 59.7 L212.4 55.7 L218.0 59.8 L223.6 60.3 L229.2 59.0 L234.8 55.3 L240.4 55.6 L246.0 56.6" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__draw" style="--len:305; --t:5s; --d:0.64s" d="M22.0 78.0 L27.6 82.4 L33.2 83.7 L38.8 79.3 L44.4 78.6 L50.0 82.2 L55.6 85.7 L61.2 73.3 L66.8 68.0 L72.4 65.0 L78.0 76.7 L83.6 73.7 L89.2 71.4 L94.8 63.6 L100.4 61.7 L106.0 62.0 L111.6 59.6 L117.2 68.5 L122.8 63.7 L128.4 62.2 L134.0 65.4 L139.6 59.3 L145.2 51.2 L150.8 57.7 L156.4 60.8 L162.0 59.5 L167.6 58.7 L173.2 60.6 L178.8 65.0 L184.4 55.4 L190.0 50.8 L195.6 56.4 L201.2 62.5 L206.8 54.8 L212.4 50.5 L218.0 42.4 L223.6 39.1 L229.2 43.4 L234.8 42.5 L240.4 52.5 L246.0 56.1" stroke="#22d3ee" stroke-opacity="0.55" stroke-width="1.2" stroke-linejoin="round"/>
            <path class="qs03__fade" style="--t:5s" d="M22.0 78.0 L27.6 77.6 L33.2 77.2 L38.8 76.9 L44.4 76.7 L50.0 76.5 L55.6 75.9 L61.2 75.5 L66.8 75.1 L72.4 74.8 L78.0 74.7 L83.6 74.6 L89.2 74.1 L94.8 74.1 L100.4 74.0 L106.0 73.4 L111.6 73.3 L117.2 73.0 L122.8 72.9 L128.4 73.2 L134.0 73.0 L139.6 73.2 L145.2 73.2 L150.8 72.8 L156.4 72.5 L162.0 72.6 L167.6 73.0 L173.2 72.7 L178.8 72.2 L184.4 72.3 L190.0 71.9 L195.6 71.7 L201.2 71.6 L206.8 72.0 L212.4 72.0 L218.0 71.6 L223.6 71.9 L229.2 71.3 L234.8 70.9 L240.4 71.1 L246.0 70.6" stroke="#ffffff" stroke-width="2" stroke-linejoin="round"/>
            <circle cx="22" cy="78.0" r="4.5" fill="#22d3ee" stroke="#ffffff" stroke-width="1.5"/>
            <text x="20" y="148" fill="#5d7086" font-size="9" letter-spacing="1.5">TODAY</text>
            <text x="250" y="148" fill="#5d7086" font-size="9" text-anchor="end" letter-spacing="1.5">HORIZON</text>
            <rect x="76" y="141" width="12" height="7" fill="#22d3ee" fill-opacity="0.3"/><text x="92" y="148" fill="#8ea3b8" font-size="9">5–95% band</text>
            </svg>
            <div>
              <p class="qs03__tag">Market risk · CCR · XVA</p>
              <h2>Neural SDE simulation</h2>
              <p class="qs03__what">Learn drift and volatility from history, then simulate risk factor paths on GPU.</p>
              <span class="qs03__model">Neural SDE · JAX</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#4da3ff">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Rough-vol parameters feed a neural pricer; each calibration iteration closes on the market smile">
            <path d="M120 14 V124 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <rect class="qs03__bar" style="--d:0.0s" x="22" y="54" width="13" height="58" rx="2" fill="#4da3ff" fill-opacity="0.9"/>
            <rect class="qs03__bar" style="--d:0.1s" x="44" y="78" width="13" height="34" rx="2" fill="#4da3ff" fill-opacity="0.7"/>
            <rect class="qs03__bar" style="--d:0.2s" x="66" y="66" width="13" height="46" rx="2" fill="#4da3ff" fill-opacity="0.5"/>
            <text x="28.5" y="126" fill="#cfd8e3" font-size="10" text-anchor="middle">H</text>
            <text x="50.5" y="126" fill="#cfd8e3" font-size="10" text-anchor="middle">η</text>
            <text x="72.5" y="126" fill="#cfd8e3" font-size="10" text-anchor="middle">ρ</text>
            <text x="18" y="12" fill="#5d7086" font-size="9" letter-spacing="1.5">PARAMS</text>
            <rect x="88" y="58" width="24" height="18" rx="4" fill="#12171e" stroke="#4da3ff" stroke-width="1.2"/>
            <text x="100" y="70" fill="#cfe5ff" font-size="8" font-weight="700" text-anchor="middle">NN</text>
            <path d="M80 67 H86 M112 67 H118" stroke="#5d7086" stroke-width="1.2"/>
            <path class="qs03__draw" style="--len:133; --d:0.00s" d="M124.0 68.0 L128.0 70.5 L132.0 72.9 L136.0 75.1 L140.0 77.1 L144.0 79.0 L148.0 80.6 L152.0 82.2 L156.0 83.5 L160.0 84.6 L164.0 85.6 L168.0 86.4 L172.0 87.1 L176.0 87.6 L180.0 87.9 L184.0 88.0 L188.0 88.0 L192.0 87.7 L196.0 87.4 L200.0 86.8 L204.0 86.1 L208.0 85.2 L212.0 84.1 L216.0 82.8 L220.0 81.4 L224.0 79.8 L228.0 78.1 L232.0 76.1 L236.0 74.0 L240.0 71.8 L244.0 69.3 L248.0 66.7" stroke="#3b4a5c" stroke-width="1.3" stroke-linecap="round"/>
            <path class="qs03__draw" style="--len:148; --d:0.35s" d="M124.0 52.9 L128.0 57.6 L132.0 62.0 L136.0 66.1 L140.0 70.0 L144.0 73.5 L148.0 76.7 L152.0 79.7 L156.0 82.3 L160.0 84.6 L164.0 86.7 L168.0 88.4 L172.0 89.9 L176.0 91.0 L180.0 91.9 L184.0 92.4 L188.0 92.7 L192.0 92.6 L196.0 92.3 L200.0 91.6 L204.0 90.7 L208.0 89.4 L212.0 87.9 L216.0 86.0 L220.0 83.9 L224.0 81.5 L228.0 78.7 L232.0 75.7 L236.0 72.4 L240.0 68.7 L244.0 64.8 L248.0 60.6" stroke="#4b5f78" stroke-width="1.3" stroke-linecap="round"/>
            <path class="qs03__draw" style="--len:162; --d:0.70s" d="M124.0 41.7 L128.0 48.1 L132.0 54.0 L136.0 59.6 L140.0 64.7 L144.0 69.5 L148.0 73.9 L152.0 77.9 L156.0 81.4 L160.0 84.6 L164.0 87.5 L168.0 89.9 L172.0 91.9 L176.0 93.5 L180.0 94.8 L184.0 95.6 L188.0 96.1 L192.0 96.2 L196.0 95.9 L200.0 95.2 L204.0 94.1 L208.0 92.6 L212.0 90.7 L216.0 88.4 L220.0 85.7 L224.0 82.7 L228.0 79.2 L232.0 75.4 L236.0 71.2 L240.0 66.5 L244.0 61.5 L248.0 56.1" stroke="#6f8fb8" stroke-width="1.3" stroke-linecap="round"/>
            <path class="qs03__draw" style="--len:173; --d:1.05s" d="M124.0 34.3 L128.0 41.7 L132.0 48.7 L136.0 55.2 L140.0 61.2 L144.0 66.8 L148.0 72.0 L152.0 76.6 L156.0 80.9 L160.0 84.6 L164.0 88.0 L168.0 90.8 L172.0 93.3 L176.0 95.2 L180.0 96.7 L184.0 97.8 L188.0 98.4 L192.0 98.6 L196.0 98.3 L200.0 97.5 L204.0 96.3 L208.0 94.6 L212.0 92.5 L216.0 90.0 L220.0 87.0 L224.0 83.5 L228.0 79.6 L232.0 75.2 L236.0 70.3 L240.0 65.1 L244.0 59.3 L248.0 53.1" stroke="#4da3ff" stroke-width="2.4" stroke-linecap="round"/>
            <circle cx="132" cy="48.7" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="150" cy="74.4" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="168" cy="90.8" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="186" cy="98.2" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="204" cy="96.3" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="222" cy="85.3" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <circle cx="240" cy="65.1" r="3.4" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/>
            <text x="124" y="12" fill="#5d7086" font-size="9" letter-spacing="1.5">IMPLIED VOL</text>
            <text x="250" y="136" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">STRIKE</text>
            <circle cx="24" cy="146" r="3" fill="#0d1117" stroke="#ffffff" stroke-width="1.5"/><text x="31" y="149" fill="#8ea3b8" font-size="9">market</text>
            <path d="M72 146 H84" stroke="#4b5f78" stroke-width="1.3"/><text x="88" y="149" fill="#8ea3b8" font-size="9">iterations</text>
            <path d="M140 146 H152" stroke="#4da3ff" stroke-width="2.4"/><text x="156" y="149" fill="#8ea3b8" font-size="9">calibrated, in ms</text>
            </svg>
            <div>
              <p class="qs03__tag">Trading · Options</p>
              <h2>Deep calibration</h2>
              <p class="qs03__what">Learn the pricer once, then calibrate rough-vol models to the surface in milliseconds.</p>
              <span class="qs03__model">Neural pricer · PyTorch</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#d29922">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Desks plotted on the FRTB P&amp;L attribution test zones; one desk is forecast to drift from green to amber">
            <rect x="30" y="14" width="216" height="110" fill="#f85149" fill-opacity="0.10"/>
            <rect x="84.0" y="41.5" width="162.0" height="82.5" fill="#d29922" fill-opacity="0.16"/>
            <rect x="138.0" y="62.1" width="108.0" height="61.9" fill="#3fb950" fill-opacity="0.18"/>
            <path d="M30 14 V124 H246" stroke="#3b4a5c" stroke-width="1.5"/>
            <text x="242" y="120" fill="#3fb950" font-size="8" font-weight="700" letter-spacing="1" text-anchor="end">GREEN</text>
            <text x="242" y="52.5" fill="#d29922" font-size="8" font-weight="700" letter-spacing="1" text-anchor="end">AMBER</text>
            <text x="242" y="25" fill="#f85149" font-size="8" font-weight="700" letter-spacing="1" text-anchor="end">RED</text>
            <circle cx="208.2" cy="99.9" r="3.6" fill="#ffffff" fill-opacity="0.85"/>
            <circle cx="181.2" cy="86.2" r="3.6" fill="#ffffff" fill-opacity="0.85"/>
            <circle cx="221.7" cy="75.9" r="3.6" fill="#ffffff" fill-opacity="0.85"/>
            <circle cx="170.4" cy="106.8" r="3.6" fill="#ffffff" fill-opacity="0.85"/>
            <circle cx="197.4" cy="69.0" r="3.6" fill="#ffffff" fill-opacity="0.85"/>
            <circle cx="156.9" cy="75.9" r="4" fill="#ffffff"/>
            <path class="qs03__draw" style="--len:49" d="M156.9 75.9 L113.7 52.5" stroke="#d29922" stroke-width="1.6"/>
            <circle class="qs03__pop" style="--d:0.5s" cx="113.7" cy="52.5" r="5" fill="#0d1117" stroke="#d29922" stroke-width="2"/>
            <text x="84.0" y="134" fill="#5d7086" font-size="8" text-anchor="middle">0.7</text>
            <text x="138.0" y="134" fill="#5d7086" font-size="8" text-anchor="middle">0.8</text>
            <text x="246" y="134" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">SPEARMAN</text>
            <text x="34" y="12" fill="#5d7086" font-size="9" letter-spacing="1.5">KS</text>
            <circle cx="34" cy="146" r="3.2" fill="#ffffff"/><text x="41" y="149" fill="#8ea3b8" font-size="9">desk today</text>
            <circle cx="104" cy="146" r="3.6" fill="#0d1117" stroke="#d29922" stroke-width="1.8"/><text x="111" y="149" fill="#8ea3b8" font-size="9">forecast next quarter</text>
            </svg>
            <div>
              <p class="qs03__tag">Market risk · FRTB</p>
              <h2>FRTB support models</h2>
              <p class="qs03__what">See which desks are drifting towards a P&amp;L attribution fail, before the quarter ends.</p>
              <span class="qs03__model">Gradient boosting · LightGBM</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#3fb950">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="A learned price fitted through noisy pathwise payoff samples, with learned deltas shown as tangents">
            <path d="M16 12 V128 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <circle cx="122.2" cy="82.1" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="125.2" cy="100.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="134.8" cy="101.5" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="135.7" cy="104.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="162.4" cy="97.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="88.6" cy="113.1" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="40.5" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="29.5" cy="100.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="242.0" cy="51.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="159.1" cy="92.9" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="55.6" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="33.5" cy="123.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="63.0" cy="121.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="124.8" cy="116.8" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="119.6" cy="124.5" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="164.7" cy="79.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="132.9" cy="101.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="82.9" cy="105.9" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="245.5" cy="55.2" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="180.0" cy="82.6" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="91.3" cy="120.9" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="35.9" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="193.2" cy="49.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="107.4" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="236.5" cy="35.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="67.4" cy="120.5" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="225.7" cy="10.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="109.8" cy="124.6" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="36.5" cy="103.8" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="81.0" cy="101.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="39.7" cy="102.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="191.3" cy="108.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="46.7" cy="121.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="33.5" cy="128.6" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="200.1" cy="75.5" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="121.1" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="63.1" cy="120.2" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="165.5" cy="85.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="46.3" cy="112.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="81.0" cy="125.1" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="239.4" cy="36.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="220.0" cy="37.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="67.6" cy="97.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="165.1" cy="112.1" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="42.7" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="78.4" cy="119.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="194.6" cy="65.7" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="36.6" cy="130.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="40.4" cy="111.9" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="155.9" cy="94.0" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="104.0" cy="81.4" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="129.3" cy="122.6" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="149.9" cy="109.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <circle cx="54.8" cy="114.3" r="1.9" fill="#8ea3b8" fill-opacity="0.55"/>
            <path class="qs03__draw" style="--len:256" d="M20.0 121.8 L23.8 121.8 L27.5 121.8 L31.3 121.7 L35.1 121.7 L38.8 121.6 L42.6 121.6 L46.4 121.5 L50.1 121.5 L53.9 121.4 L57.7 121.3 L61.4 121.1 L65.2 121.0 L69.0 120.9 L72.7 120.7 L76.5 120.5 L80.3 120.2 L84.0 120.0 L87.8 119.6 L91.6 119.3 L95.3 118.9 L99.1 118.4 L102.9 117.9 L106.6 117.2 L110.4 116.6 L114.2 115.8 L117.9 114.9 L121.7 113.9 L125.5 112.9 L129.2 111.7 L133.0 110.4 L136.8 108.9 L140.5 107.3 L144.3 105.6 L148.1 103.8 L151.8 101.8 L155.6 99.7 L159.4 97.5 L163.1 95.2 L166.9 92.7 L170.7 90.1 L174.4 87.5 L178.2 84.7 L182.0 81.8 L185.7 78.9 L189.5 75.9 L193.3 72.8 L197.0 69.7 L200.8 66.5 L204.6 63.3 L208.3 60.0 L212.1 56.7 L215.9 53.4 L219.6 50.0 L223.4 46.7 L227.2 43.3 L230.9 39.8 L234.7 36.4 L238.5 32.9 L242.2 29.5 L246.0 26.0" stroke="#4da3ff" stroke-width="2.4" stroke-linecap="round"/>
            <g class="qs03__pop" style="--d:0.30s"><path d="M90.1 120.0 L121.7 114.7" stroke="#3fb950" stroke-width="2.2" stroke-linecap="round"/><circle cx="105.9" cy="117.4" r="2.6" fill="#3fb950"/></g>
            <g class="qs03__pop" style="--d:0.42s"><path d="M129.8 112.4 L158.8 98.8" stroke="#3fb950" stroke-width="2.2" stroke-linecap="round"/><circle cx="144.3" cy="105.6" r="2.6" fill="#3fb950"/></g>
            <g class="qs03__pop" style="--d:0.54s"><path d="M165.4 94.3 L191.0 75.1" stroke="#3fb950" stroke-width="2.2" stroke-linecap="round"/><circle cx="178.2" cy="84.7" r="2.6" fill="#3fb950"/></g>
            <g class="qs03__pop" style="--d:0.66s"><path d="M202.4 65.3 L226.3 44.1" stroke="#3fb950" stroke-width="2.2" stroke-linecap="round"/><circle cx="214.4" cy="54.7" r="2.6" fill="#3fb950"/></g>
            <text x="20" y="10" fill="#5d7086" font-size="9" letter-spacing="1.5">PRICE</text>
            <text x="250" y="140" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">SPOT</text>
            <circle cx="24" cy="146" r="2.4" fill="#8ea3b8"/><text x="31" y="149" fill="#8ea3b8" font-size="9">pathwise samples</text>
            <path d="M110 146 H122" stroke="#4da3ff" stroke-width="2.4"/><text x="126" y="149" fill="#8ea3b8" font-size="9">price</text>
            <path d="M160 146 H172" stroke="#3fb950" stroke-width="2.2"/><text x="176" y="149" fill="#8ea3b8" font-size="9">delta</text>
            </svg>
            <div>
              <p class="qs03__tag">Front office · XVA</p>
              <h2>Differential ML surrogates</h2>
              <p class="qs03__what">Train on prices and their AAD sensitivities for fast, consistent greeks.</p>
              <span class="qs03__model">Differential network · JAX</span>
            </div>
          </section>
        </div>
        <div class="qs03__page">
          <section class="qs03__card" style="--c:#f85149">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="A gradient search walks from today to the most plausible scenario that breaches a loss limit">
            <path d="M16 12 V128 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <path d="M132 12 Q166 76 248 98 L250 98 V12 Z" fill="#f85149" fill-opacity="0.10"/>
            <path d="M132 12 Q166 76 248 98" stroke="#f85149" stroke-width="2"/>
            <text x="244" y="24" fill="#f85149" font-size="8" font-weight="700" letter-spacing="1" text-anchor="end">LOSS &gt; LIMIT</text>
            <ellipse cx="84" cy="90" rx="20" ry="11" transform="rotate(-24 84 90)" stroke="#8ea3b8" stroke-opacity="0.7" stroke-width="1.1" stroke-dasharray="3 3"/>
            <ellipse cx="84" cy="90" rx="42" ry="23" transform="rotate(-24 84 90)" stroke="#8ea3b8" stroke-opacity="0.5" stroke-width="1.1" stroke-dasharray="3 3"/>
            <ellipse cx="84" cy="90" rx="66" ry="36" transform="rotate(-24 84 90)" stroke="#8ea3b8" stroke-opacity="0.35" stroke-width="1.1" stroke-dasharray="3 3"/>
            <ellipse cx="84" cy="90" rx="92" ry="50" transform="rotate(-24 84 90)" stroke="#8ea3b8" stroke-opacity="0.22" stroke-width="1.1" stroke-dasharray="3 3"/>
            <path class="qs03__draw" style="--len:93" d="M84.0 90.0 L106.9 87.2 L127.5 83.0 L144.5 76.7 L157.9 68.2 L169.0 58.4" stroke="#ffffff" stroke-width="1.4" stroke-opacity="0.8"/>
            <circle class="qs03__pop" style="--d:0.00s" cx="106.9" cy="87.2" r="2.6" fill="#ffffff"/>
            <circle class="qs03__pop" style="--d:0.12s" cx="127.5" cy="83.0" r="2.6" fill="#ffffff"/>
            <circle class="qs03__pop" style="--d:0.24s" cx="144.5" cy="76.7" r="2.6" fill="#ffffff"/>
            <circle class="qs03__pop" style="--d:0.36s" cx="157.9" cy="68.2" r="2.6" fill="#ffffff"/>
            <circle cx="84" cy="90" r="3.6" fill="#4da3ff" stroke="#ffffff" stroke-width="1.2"/>
            <circle class="qs03__pop" style="--d:0.7s" cx="169.0" cy="58.4" r="5.2" fill="#f85149" stroke="#ffffff" stroke-width="1.6"/>
            <text x="84" y="104" fill="#8ea3b8" font-size="8" text-anchor="middle">today</text>
            <text x="20" y="10" fill="#5d7086" font-size="9" letter-spacing="1.5">EQUITY</text>
            <text x="250" y="140" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">RATES</text>
            <ellipse cx="26" cy="146" rx="6" ry="3.5" stroke="#8ea3b8" stroke-width="1.1" stroke-dasharray="2 2"/><text x="36" y="149" fill="#8ea3b8" font-size="9">plausibility</text>
            <path d="M96 146 H108" stroke="#f85149" stroke-width="2"/><text x="112" y="149" fill="#8ea3b8" font-size="9">loss limit</text>
            <circle cx="166" cy="146" r="3.4" fill="#f85149" stroke="#ffffff" stroke-width="1.2"/><text x="173" y="149" fill="#8ea3b8" font-size="9">scenario found</text>
            </svg>
            <div>
              <p class="qs03__tag">Market risk · Stress testing</p>
              <h2>Reverse stress testing</h2>
              <p class="qs03__what">Search backwards through the proxy for the most plausible scenario that breaks a limit.</p>
              <span class="qs03__model">Differentiable proxy · PyTorch</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#a78bfa">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Daily returns with two stressed regimes shaded, and the detected probability of a stressed regime below">
            <rect class="qs03__fade" style="--d:0.0s" x="108.4" y="14" width="46.2" height="114" fill="#a78bfa" fill-opacity="0.13"/>
            <rect class="qs03__fade" style="--d:0.2s" x="198.8" y="14" width="28.1" height="114" fill="#a78bfa" fill-opacity="0.13"/>
            <path d="M16 50 H250" stroke="#3b4a5c" stroke-width="1"/>
            <path d="M20.0 50 V55.9" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M23.0 50 V55.8" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M26.0 50 V46.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M29.0 50 V61.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M32.1 50 V50.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M35.1 50 V61.4" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M38.1 50 V44.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M41.1 50 V49.0" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M44.1 50 V43.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M47.1 50 V52.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M50.1 50 V48.0" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M53.1 50 V51.4" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M56.2 50 V53.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M59.2 50 V49.3" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M62.2 50 V56.3" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M65.2 50 V51.8" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M68.2 50 V46.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M71.2 50 V49.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M74.2 50 V52.1" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M77.3 50 V39.0" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M80.3 50 V49.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M83.3 50 V53.0" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M86.3 50 V49.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M89.3 50 V52.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M92.3 50 V51.9" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M95.3 50 V51.8" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M98.3 50 V39.8" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M101.4 50 V49.9" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M104.4 50 V49.1" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M107.4 50 V46.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M110.4 50 V20.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M113.4 50 V54.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M116.4 50 V61.2" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M119.4 50 V20.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M122.5 50 V76.3" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M125.5 50 V56.6" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M128.5 50 V38.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M131.5 50 V20.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M134.5 50 V67.1" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M137.5 50 V80.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M140.5 50 V38.1" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M143.5 50 V59.4" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M146.6 50 V57.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M149.6 50 V41.7" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M152.6 50 V46.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M155.6 50 V48.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M158.6 50 V52.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M161.6 50 V43.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M164.6 50 V42.4" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M167.7 50 V49.8" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M170.7 50 V52.3" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M173.7 50 V46.3" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M176.7 50 V47.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M179.7 50 V55.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M182.7 50 V52.3" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M185.7 50 V44.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M188.7 50 V50.5" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M191.8 50 V51.7" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M194.8 50 V48.9" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M197.8 50 V50.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M200.8 50 V50.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M203.8 50 V80.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M206.8 50 V20.0" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M209.8 50 V46.4" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M212.9 50 V65.8" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M215.9 50 V32.4" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M218.9 50 V44.4" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M221.9 50 V49.7" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M224.9 50 V31.1" stroke="#cfd8e3" stroke-opacity="0.95" stroke-width="1.8"/>
            <path d="M227.9 50 V38.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M230.9 50 V47.2" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M233.9 50 V42.0" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M237.0 50 V39.1" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M240.0 50 V55.1" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M243.0 50 V52.6" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M246.0 50 V46.1" stroke="#8ea3b8" stroke-opacity="0.6" stroke-width="1.8"/>
            <path d="M16 96 V128 H250" stroke="#3b4a5c" stroke-width="1"/>
            <path class="qs03__draw" style="--len:338; --t:5s" d="M20.0 124.6 L23.0 124.5 L26.0 125.3 L29.0 122.2 L32.1 125.1 L35.1 122.0 L38.1 122.7 L41.1 125.2 L44.1 124.5 L47.1 125.6 L50.1 126.3 L53.1 126.8 L56.2 126.8 L59.2 127.2 L62.2 126.8 L65.2 127.0 L68.2 127.0 L71.2 127.3 L74.2 127.4 L77.3 126.2 L80.3 126.9 L83.3 127.0 L86.3 127.3 L89.3 127.3 L92.3 127.4 L95.3 127.4 L98.3 126.5 L101.4 127.1 L104.4 127.3 L107.4 127.3 L110.4 106.3 L113.4 112.7 L116.4 110.1 L119.4 96.2 L122.5 96.1 L125.5 96.3 L128.5 96.6 L131.5 96.0 L134.5 96.0 L137.5 96.0 L140.5 96.0 L143.5 96.1 L146.6 96.3 L149.6 96.8 L152.6 99.3 L155.6 106.9 L158.6 115.2 L161.6 117.0 L164.6 117.4 L167.7 123.2 L170.7 125.0 L173.7 125.5 L176.7 126.2 L179.7 126.0 L182.7 126.5 L185.7 126.2 L188.7 126.9 L191.8 127.2 L194.8 127.4 L197.8 127.5 L200.8 127.6 L203.8 112.3 L206.8 97.5 L209.8 101.3 L212.9 99.3 L215.9 97.8 L218.9 100.7 L221.9 110.4 L224.9 101.9 L227.9 101.9 L230.9 109.5 L233.9 111.1 L237.0 109.3 L240.0 114.0 L243.0 119.7 L246.0 122.2" stroke="#a78bfa" stroke-width="2.2" stroke-linejoin="round"/>
            <text x="20" y="10" fill="#5d7086" font-size="9" letter-spacing="1.5">RETURNS</text>
            <text x="20" y="92" fill="#5d7086" font-size="9" letter-spacing="1.5">P(STRESSED)</text>
            <rect x="20" y="142" width="10" height="7" fill="#a78bfa" fill-opacity="0.3"/><text x="34" y="149" fill="#8ea3b8" font-size="9">stressed regime</text>
            <path d="M110 146 H122" stroke="#a78bfa" stroke-width="2.2"/><text x="126" y="149" fill="#8ea3b8" font-size="9">detected probability</text>
            </svg>
            <div>
              <p class="qs03__tag">Market risk · Trading</p>
              <h2>Regime detection</h2>
              <p class="qs03__what">Spot a shift into a stressed regime early, to rescale VaR and review limits.</p>
              <span class="qs03__model">Hidden Markov model · JAX</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#3fb950">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Expected exposure and potential future exposure profiles over time, with surrogate predictions">
            <path d="M16 12 V128 H250" stroke="#3b4a5c" stroke-width="1.5"/>
            <path class="qs03__fade" d="M20.0 124.0 L25.6 109.6 L31.3 104.2 L37.0 100.5 L42.6 97.6 L48.2 95.4 L53.9 93.7 L59.5 92.3 L65.2 91.3 L70.8 90.5 L76.5 89.9 L82.2 89.5 L87.8 89.4 L93.5 89.4 L99.1 89.5 L104.8 89.8 L110.4 90.3 L116.0 90.8 L121.7 91.5 L127.3 92.2 L133.0 93.1 L138.7 94.1 L144.3 95.2 L149.9 96.3 L155.6 97.5 L161.2 98.8 L166.9 100.2 L172.6 101.7 L178.2 103.2 L183.8 104.7 L189.5 106.4 L195.2 108.0 L200.8 109.7 L206.4 111.5 L212.1 113.3 L217.8 115.1 L223.4 116.9 L229.1 118.8 L234.7 120.6 L240.3 122.4 L246.0 124.0 L246 128 L20 128 Z" fill="#3fb950" fill-opacity="0.12"/>
            <path class="qs03__fade" style="--d:0s" d="M20.0 124.0 L25.6 93.8 L31.3 82.5 L37.0 74.6 L42.6 68.6 L48.2 64.0 L53.9 60.3 L59.5 57.5 L65.2 55.2 L70.8 53.6 L76.5 52.4 L82.2 51.6 L87.8 51.3 L93.5 51.3 L99.1 51.6 L104.8 52.2 L110.4 53.1 L116.0 54.3 L121.7 55.7 L127.3 57.3 L133.0 59.2 L138.7 61.2 L144.3 63.4 L149.9 65.8 L155.6 68.4 L161.2 71.2 L166.9 74.1 L172.6 77.1 L178.2 80.3 L183.8 83.5 L189.5 87.0 L195.2 90.5 L200.8 94.1 L206.4 97.8 L212.1 101.5 L217.8 105.3 L223.4 109.2 L229.1 113.1 L234.7 116.9 L240.3 120.6 L246.0 124.0" stroke="#8ea3b8" stroke-width="1.5" stroke-dasharray="4 3"/>
            <path class="qs03__draw" style="--len:245; --d:0.2s" d="M20.0 124.0 L25.6 109.6 L31.3 104.2 L37.0 100.5 L42.6 97.6 L48.2 95.4 L53.9 93.7 L59.5 92.3 L65.2 91.3 L70.8 90.5 L76.5 89.9 L82.2 89.5 L87.8 89.4 L93.5 89.4 L99.1 89.5 L104.8 89.8 L110.4 90.3 L116.0 90.8 L121.7 91.5 L127.3 92.2 L133.0 93.1 L138.7 94.1 L144.3 95.2 L149.9 96.3 L155.6 97.5 L161.2 98.8 L166.9 100.2 L172.6 101.7 L178.2 103.2 L183.8 104.7 L189.5 106.4 L195.2 108.0 L200.8 109.7 L206.4 111.5 L212.1 113.3 L217.8 115.1 L223.4 116.9 L229.1 118.8 L234.7 120.6 L240.3 122.4 L246.0 124.0" stroke="#3fb950" stroke-width="2.4" stroke-linecap="round"/>
            <circle class="qs03__pop" style="--d:0.00s" cx="42.6" cy="97.6" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.08s" cx="76.5" cy="89.9" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.16s" cx="110.4" cy="90.3" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.24s" cx="144.3" cy="95.2" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.32s" cx="178.2" cy="103.2" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <circle class="qs03__pop" style="--d:0.40s" cx="212.1" cy="113.3" r="3.4" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/>
            <text x="250" y="142" fill="#5d7086" font-size="9" text-anchor="end" letter-spacing="1.5">TIME</text>
            <path d="M20 146 H32" stroke="#3fb950" stroke-width="2.4"/><text x="36" y="149" fill="#8ea3b8" font-size="9">EE</text>
            <path d="M60 146 H72" stroke="#8ea3b8" stroke-width="1.5" stroke-dasharray="4 3"/><text x="76" y="149" fill="#8ea3b8" font-size="9">PFE 97.5%</text>
            <circle cx="138" cy="146" r="3" fill="#ffffff" stroke="#3fb950" stroke-width="1.5"/><text x="145" y="149" fill="#8ea3b8" font-size="9">surrogate</text>
            </svg>
            <div>
              <p class="qs03__tag">XVA · CCR</p>
              <h2>Exposure profile surrogate</h2>
              <p class="qs03__what">Predict EE and PFE per netting set without nested Monte Carlo.</p>
              <span class="qs03__model">Gradient boosting · LightGBM</span>
            </div>
          </section>
          <section class="qs03__card" style="--c:#22d3ee">
            <svg viewBox="0 0 260 150" fill="none" role="img" aria-label="Daily unexplained P&amp;L inside a learned normal range, with three outlier days flagged">
            <rect class="qs03__fade" x="16" y="56" width="234" height="32" fill="#22d3ee" fill-opacity="0.10"/>
            <path class="qs03__fade" d="M16 56 H250 M16 88 H250" stroke="#22d3ee" stroke-opacity="0.6" stroke-width="1" stroke-dasharray="3 3"/>
            <path d="M16 72 H250" stroke="#3b4a5c" stroke-width="1"/>
            <rect class="qs03__bar" style="--d:0.00s; transform-origin: top" x="21.0" y="72.0" width="4.6" height="8.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.04s; transform-origin: bottom" x="27.6" y="69.9" width="4.6" height="2.1" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.08s; transform-origin: bottom" x="34.3" y="59.8" width="4.6" height="12.2" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.12s; transform-origin: bottom" x="40.9" y="57.5" width="4.6" height="14.5" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.16s; transform-origin: bottom" x="47.6" y="60.8" width="4.6" height="11.2" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.20s; transform-origin: bottom" x="54.2" y="71.5" width="4.6" height="0.5" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.24s; transform-origin: bottom" x="60.9" y="68.9" width="4.6" height="3.1" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.28s; transform-origin: top" x="67.5" y="72.0" width="4.6" height="2.3" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.32s; transform-origin: top" x="74.2" y="72.0" width="4.6" height="12.3" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.36s; transform-origin: bottom" x="80.8" y="26.1" width="4.6" height="45.9" rx="1" fill="#f85149" fill-opacity="1"/>
            <circle class="qs03__pop" style="--d:0.66s" cx="83.1" cy="20.1" r="4.5" stroke="#f85149" stroke-width="1.6"/>
            <rect class="qs03__bar" style="--d:0.40s; transform-origin: bottom" x="87.5" y="63.1" width="4.6" height="8.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.44s; transform-origin: bottom" x="94.1" y="69.7" width="4.6" height="2.3" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.48s; transform-origin: top" x="100.8" y="72.0" width="4.6" height="3.0" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.52s; transform-origin: top" x="107.4" y="72.0" width="4.6" height="9.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.56s; transform-origin: top" x="114.1" y="72.0" width="4.6" height="0.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.60s; transform-origin: top" x="120.7" y="72.0" width="4.6" height="5.5" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.64s; transform-origin: bottom" x="127.4" y="71.7" width="4.6" height="0.3" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.68s; transform-origin: bottom" x="134.0" y="67.4" width="4.6" height="4.6" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.72s; transform-origin: top" x="140.6" y="72.0" width="4.6" height="21.4" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.76s; transform-origin: top" x="147.3" y="72.0" width="4.6" height="0.8" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.80s; transform-origin: top" x="153.9" y="72.0" width="4.6" height="13.7" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.84s; transform-origin: top" x="160.6" y="72.0" width="4.6" height="51.3" rx="1" fill="#f85149" fill-opacity="1"/>
            <circle class="qs03__pop" style="--d:1.14s" cx="162.9" cy="129.3" r="4.5" stroke="#f85149" stroke-width="1.6"/>
            <rect class="qs03__bar" style="--d:0.88s; transform-origin: top" x="167.2" y="72.0" width="4.6" height="11.6" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.92s; transform-origin: bottom" x="173.9" y="61.6" width="4.6" height="10.4" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:0.96s; transform-origin: bottom" x="180.5" y="65.7" width="4.6" height="6.3" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.00s; transform-origin: bottom" x="187.2" y="67.3" width="4.6" height="4.7" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.04s; transform-origin: top" x="193.8" y="72.0" width="4.6" height="7.0" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.08s; transform-origin: top" x="200.5" y="72.0" width="4.6" height="13.5" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.12s; transform-origin: bottom" x="207.1" y="31.5" width="4.6" height="40.5" rx="1" fill="#f85149" fill-opacity="1"/>
            <circle class="qs03__pop" style="--d:1.42s" cx="209.4" cy="25.5" r="4.5" stroke="#f85149" stroke-width="1.6"/>
            <rect class="qs03__bar" style="--d:1.16s; transform-origin: bottom" x="213.8" y="70.1" width="4.6" height="1.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.20s; transform-origin: top" x="220.4" y="72.0" width="4.6" height="0.7" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.24s; transform-origin: bottom" x="227.1" y="65.5" width="4.6" height="6.5" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.28s; transform-origin: top" x="233.7" y="72.0" width="4.6" height="3.4" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <rect class="qs03__bar" style="--d:1.32s; transform-origin: bottom" x="240.4" y="55.1" width="4.6" height="16.9" rx="1" fill="#8ea3b8" fill-opacity="0.55"/>
            <text x="20" y="10" fill="#5d7086" font-size="9" letter-spacing="1.5">UNEXPLAINED P&amp;L</text>
            <text x="250" y="136" fill="#5d7086" font-size="9" letter-spacing="1.5" text-anchor="end">DAYS</text>
            <rect x="20" y="142" width="10" height="7" fill="#22d3ee" fill-opacity="0.3"/><text x="34" y="149" fill="#8ea3b8" font-size="9">learned normal range</text>
            <rect x="132" y="142" width="7" height="7" rx="1" fill="#f85149"/><text x="143" y="149" fill="#8ea3b8" font-size="9">flagged</text>
            </svg>
            <div>
              <p class="qs03__tag">Risk controls · P&amp;L</p>
              <h2>P&amp;L anomaly detection</h2>
              <p class="qs03__what">Catch unexplained P&amp;L and bad market data before they reach the reports.</p>
              <span class="qs03__model">Isolation forest · scikit-learn</span>
            </div>
          </section>
        </div>
      </div>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>03</span></div>
  </div>
</div>
