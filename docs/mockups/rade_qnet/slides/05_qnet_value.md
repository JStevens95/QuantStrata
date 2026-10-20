<!--
  QNet deck - 05 What it means for you.

  Message: one line of value for each audience in the room, with the
  flagship model as the proof that the platform holds on a hard problem.

  The examples under each audience are where QNet fits, not models that
  ship with it; anything not yet built is tagged "coming". The HTML has
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
  @media (prefers-reduced-motion: reduce) { .qs *, .qs *::after { animation: none !important; } }

  .qs05__stage { display: flex; flex-direction: column; justify-content: center; gap: 1.6cqw; }
  .qs05__cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.6cqw; }
  .qs05__card {
    padding: 1.8cqw 2cqw; border: 1px solid #2a3340; border-radius: 1.2cqw;
    background: rgba(22, 27, 34, 0.85); border-top: 0.3cqw solid var(--c);
  }
  .qs05__card svg { width: 3cqw; height: 3cqw; }
  .qs05__card h2 { margin: 0.9cqw 0 0; font-size: 1.05cqw; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--c); }
  .qs05__card h3 { margin: 0.5cqw 0 0; font-size: 1.75cqw; line-height: 1.2; font-weight: 700; }
  .qs05__card ul { margin: 1.1cqw 0 0; padding: 0; list-style: none; display: grid; gap: 0.5cqw; }
  .qs05__card li { font-size: 1.15cqw; color: #8ea3b8; padding-left: 1.2cqw; position: relative; }
  .qs05__card li::before { content: ""; position: absolute; left: 0; top: 0.6cqw; width: 0.45cqw; height: 0.45cqw; border-radius: 50%; background: var(--c); }
  .qs05__soon {
    margin-left: 0.5cqw; padding: 0.1cqw 0.6cqw; border: 1px solid #5d7086; border-radius: 999px;
    font-size: 0.8cqw; letter-spacing: 0.1em; text-transform: uppercase; color: #8ea3b8;
  }
  .qs05__proof {
    display: flex; align-items: center; gap: 2cqw; padding: 1.2cqw 2cqw;
    border: 1px solid #24502f; border-radius: 1.2cqw;
    background: linear-gradient(90deg, rgba(63, 185, 80, 0.10), rgba(22, 27, 34, 0.6));
  }
  .qs05__proof svg { width: 13cqw; flex: none; }
  .qs05__proof b { display: block; font-size: 1.55cqw; }
  .qs05__proof span { display: block; margin-top: 0.3cqw; font-size: 1.15cqw; color: #8ea3b8; }
  .qs05__spark { stroke-dasharray: 320; stroke-dashoffset: 320; animation: qs05-draw 1.6s ease-out 1.2s forwards; }
  @keyframes qs05-draw { to { stroke-dashoffset: 0; } }
  .qs05__node { animation: qs05-glow 2.4s ease-in-out infinite; animation-delay: var(--d); }
  @keyframes qs05-glow { 0%, 100% { fill: #4da3ff; } 50% { fill: #cfe5ff; } }
</style>

<div class="qs qs05">
  <div class="qs__frame">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs05-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs05-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs05-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs05-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs05-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs05-brand-body)"/>
    <g clip-path="url(#qs05-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs05-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs05-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
    <path id="qs05-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs05-brand-signal)" stroke-width="2.4"/>
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
    <circle cx="128" cy="124" r="37" fill="url(#qs05-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">05 · What it means for you</p>
    <h1 class="qs__title qs-in" style="--i:1">Less plumbing. More models you can defend.</h1>
    <div class="qs__stage qs05__stage">
      <div class="qs05__cards">
        <section class="qs05__card qs-in" style="--i:2; --c:#4da3ff">
          <svg viewBox="0 0 24 24" fill="none" stroke="#4da3ff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 L20 6 V12 C20 16.5 16.5 19.8 12 21 C7.5 19.8 4 16.5 4 12 V6 Z"/><path d="M8.5 12 L11 14.5 L15.5 9.5"/></svg>
          <h2>Risk quants</h2>
          <h3>Validation evidence, built in</h3>
          <ul>
            <li>P&amp;L replication and explain</li>
            <li>Rebuild any result from its bundle</li>
          </ul>
        </section>
        <section class="qs05__card qs-in" style="--i:3; --c:#3fb950">
          <svg viewBox="0 0 24 24" fill="none" stroke="#3fb950" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17 L9 11 L13 15 L21 7"/><path d="M15 7 H21 V13"/></svg>
          <h2>Front office</h2>
          <h3>From idea to served model, one path</h3>
          <ul>
            <li>Fast pricing and greeks surrogates</li>
            <li>Hedging policies<span class="qs05__soon">coming</span></li>
          </ul>
        </section>
        <section class="qs05__card qs-in" style="--i:4; --c:#a78bfa">
          <svg viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>
          <h2>XVA</h2>
          <h3>Scale to every netting set</h3>
          <ul>
            <li>Exposure surrogates per netting set</li>
            <li>One spec, run in parallel</li>
          </ul>
        </section>
      </div>
      <div class="qs05__proof qs-in" style="--i:6">
        <svg viewBox="0 0 220 90" fill="none" role="img" aria-label="A graph of instruments above a P&amp;L series">
          <g stroke="#ffffff" stroke-opacity="0.35" stroke-width="1.5">
            <path d="M20 22 L62 12 L104 28 L146 14 L190 26"/>
            <path d="M20 22 L104 28 L190 26"/>
            <path d="M62 12 L146 14"/>
          </g>
          <circle class="qs05__node" style="--d:0s" cx="20" cy="22" r="5" fill="#4da3ff"/>
          <circle class="qs05__node" style="--d:0.4s" cx="62" cy="12" r="5" fill="#4da3ff"/>
          <circle class="qs05__node" style="--d:0.8s" cx="104" cy="28" r="5" fill="#4da3ff"/>
          <circle class="qs05__node" style="--d:1.2s" cx="146" cy="14" r="5" fill="#4da3ff"/>
          <circle class="qs05__node" style="--d:1.6s" cx="190" cy="26" r="5" fill="#4da3ff"/>
          <path class="qs05__spark" d="M10 70 L30 62 L48 74 L66 56 L84 64 L102 50 L120 66 L138 58 L156 46 L174 60 L192 52 L210 44" stroke="#3fb950" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div><b>Proven on the hardest case first</b><span>A hybrid graph-temporal network that replicates desk P&amp;L runs on QNet today, matching the original implementation.</span></div>
      </div>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>05</span></div>
  </div>
</div>
