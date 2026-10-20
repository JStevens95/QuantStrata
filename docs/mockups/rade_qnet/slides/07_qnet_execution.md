<!--
  QNet deck - 07 Execution and placement.

  Message: one specification expands into many independent jobs, and
  where they run is configuration. Sequential, CPU processes, one worker
  per GPU, or a job set spread across machines. Every mode ends at the
  same 39 of 40, because placement cannot change the results.

  Click a placement card to re-run the animation in that mode. The cards
  are labels for hidden radio buttons, so it works with no script. Each
  mode has its own keyframe names so that switching restarts the run.
  The HTML has no blank lines on purpose: Markdown ends an HTML block at
  the first blank line.
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

  .qs07__mode { position: absolute; opacity: 0; pointer-events: none; }
  .qs07__stage { display: flex; flex-direction: column; gap: 1.2cqw; }
  .qs07__diagram { width: 100%; flex: 1; min-height: 0; }
  .qs07__diagram text { font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
  .qs07__cell { fill: #161b22; stroke: #2a3340; stroke-width: 1.5; }
  .qs07__lane, .qs07__placement { display: none; }
  .qs07__late { opacity: 0; }
  @keyframes qs07-done-l { from { fill: #161b22; stroke: #2a3340; } to { fill: #173d24; stroke: #3fb950; } }
  @keyframes qs07-done-p { from { fill: #161b22; stroke: #2a3340; } to { fill: #173d24; stroke: #3fb950; } }
  @keyframes qs07-done-g { from { fill: #161b22; stroke: #2a3340; } to { fill: #173d24; stroke: #3fb950; } }
  @keyframes qs07-done-j { from { fill: #161b22; stroke: #2a3340; } to { fill: #173d24; stroke: #3fb950; } }
  @keyframes qs07-fail-l { from { fill: #161b22; stroke: #2a3340; } to { fill: #3d2f12; stroke: #d29922; } }
  @keyframes qs07-fail-p { from { fill: #161b22; stroke: #2a3340; } to { fill: #3d2f12; stroke: #d29922; } }
  @keyframes qs07-fail-g { from { fill: #161b22; stroke: #2a3340; } to { fill: #3d2f12; stroke: #d29922; } }
  @keyframes qs07-fail-j { from { fill: #161b22; stroke: #2a3340; } to { fill: #3d2f12; stroke: #d29922; } }
  @keyframes qs07-late-l { from { opacity: 0; } to { opacity: 1; } }
  @keyframes qs07-late-p { from { opacity: 0; } to { opacity: 1; } }
  @keyframes qs07-late-g { from { opacity: 0; } to { opacity: 1; } }
  @keyframes qs07-late-j { from { opacity: 0; } to { opacity: 1; } }
  #qs07-local:checked ~ .qs__stage .qs07__cell { animation: qs07-done-l 0.3s ease both; animation-delay: var(--l); }
  #qs07-local:checked ~ .qs__stage .qs07__cell.is-fail { animation-name: qs07-fail-l; }
  #qs07-local:checked ~ .qs__stage .qs07__late { animation: qs07-late-l 0.5s ease 3.4s both; }
  #qs07-local:checked ~ .qs__stage .qs07__lane--local,
  #qs07-local:checked ~ .qs__stage .qs07__placement--local { display: inline; }
  #qs07-local:checked ~ .qs__stage .qs07__exec--local { border-color: #3fb950; box-shadow: 0 0 1.6cqw rgba(63, 185, 80, 0.25); }
  #qs07-procs:checked ~ .qs__stage .qs07__cell { animation: qs07-done-p 0.3s ease both; animation-delay: var(--p); }
  #qs07-procs:checked ~ .qs__stage .qs07__cell.is-fail { animation-name: qs07-fail-p; }
  #qs07-procs:checked ~ .qs__stage .qs07__late { animation: qs07-late-p 0.5s ease 2.6s both; }
  #qs07-procs:checked ~ .qs__stage .qs07__lane--procs,
  #qs07-procs:checked ~ .qs__stage .qs07__placement--procs { display: inline; }
  #qs07-procs:checked ~ .qs__stage .qs07__exec--procs { border-color: #3fb950; box-shadow: 0 0 1.6cqw rgba(63, 185, 80, 0.25); }
  #qs07-gpus:checked ~ .qs__stage .qs07__cell { animation: qs07-done-g 0.3s ease both; animation-delay: var(--g); }
  #qs07-gpus:checked ~ .qs__stage .qs07__cell.is-fail { animation-name: qs07-fail-g; }
  #qs07-gpus:checked ~ .qs__stage .qs07__late { animation: qs07-late-g 0.5s ease 2.9s both; }
  #qs07-gpus:checked ~ .qs__stage .qs07__lane--gpus,
  #qs07-gpus:checked ~ .qs__stage .qs07__placement--gpus { display: inline; }
  #qs07-gpus:checked ~ .qs__stage .qs07__exec--gpus { border-color: #3fb950; box-shadow: 0 0 1.6cqw rgba(63, 185, 80, 0.25); }
  #qs07-jobset:checked ~ .qs__stage .qs07__cell { animation: qs07-done-j 0.3s ease both; animation-delay: var(--j); }
  #qs07-jobset:checked ~ .qs__stage .qs07__cell.is-fail { animation-name: qs07-fail-j; }
  #qs07-jobset:checked ~ .qs__stage .qs07__late { animation: qs07-late-j 0.5s ease 1.6s both; }
  #qs07-jobset:checked ~ .qs__stage .qs07__lane--jobset,
  #qs07-jobset:checked ~ .qs__stage .qs07__placement--jobset { display: inline; }
  #qs07-jobset:checked ~ .qs__stage .qs07__exec--jobset { border-color: #3fb950; box-shadow: 0 0 1.6cqw rgba(63, 185, 80, 0.25); }
  .qs07__executors { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.2cqw; }
  .qs07__exec {
    display: block; padding: 0.9cqw 1.2cqw; border: 1px solid #2a3340; border-radius: 0.9cqw; background: #161b22;
    cursor: pointer; user-select: none; transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
  }
  .qs07__exec:hover { background: #1b222c; }
  .qs07__exec b { display: block; font-size: 1.3cqw; }
  .qs07__exec span { display: block; margin-top: 0.2cqw; font-size: 1cqw; color: #8ea3b8; }
  .qs07__hint { margin: 0; font-size: 0.9cqw; letter-spacing: 0.14em; text-transform: uppercase; color: #5d7086; text-align: right; }
</style>

<div class="qs qs07">
  <div class="qs__frame">
    <input type="radio" name="qs07-mode" id="qs07-local" class="qs07__mode">
    <input type="radio" name="qs07-mode" id="qs07-procs" class="qs07__mode" checked>
    <input type="radio" name="qs07-mode" id="qs07-gpus" class="qs07__mode">
    <input type="radio" name="qs07-mode" id="qs07-jobset" class="qs07__mode">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs07-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs07-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs07-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs07-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs07-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs07-brand-body)"/>
    <g clip-path="url(#qs07-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs07-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs07-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
    <path id="qs07-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs07-brand-signal)" stroke-width="2.4"/>
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
    <circle cx="128" cy="124" r="37" fill="url(#qs07-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">07 · Execution and placement</p>
    <h1 class="qs__title qs-in" style="--i:1">One spec. Many jobs. Any hardware.</h1>
    <p class="qs__sub qs-in" style="--i:2">Placement is configuration. It can never change the results.</p>
    <div class="qs__stage qs07__stage">
      <svg class="qs07__diagram qs-in" style="--i:3" viewBox="0 0 1200 400" fill="none" role="img" aria-label="One specification fans out to forty jobs; thirty-nine train and one failure is recorded, whichever placement runs them">
        <rect x="30" y="80" width="300" height="240" rx="16" fill="#161b22" stroke="#2d5680" stroke-width="1.5"/>
        <rect x="30" y="80" width="300" height="48" rx="16" fill="#1b2a3d"/>
        <rect x="30" y="112" width="300" height="16" fill="#1b2a3d"/>
        <text x="54" y="111" fill="#cfe5ff" font-size="17" font-weight="600" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">job_set.yaml</text>
        <g font-size="18" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
          <text x="54" y="178"><tspan fill="#4da3ff">model:</tspan><tspan fill="#e8eef5"> any_model</tspan></text>
          <text x="54" y="228"><tspan fill="#4da3ff">jobs:</tspan><tspan fill="#e8eef5"> 40</tspan></text>
          <text x="54" y="278" fill="#4da3ff">placement:</text>
          <text class="qs07__placement qs07__placement--local" x="160" y="278" fill="#e8eef5">local</text>
          <text class="qs07__placement qs07__placement--procs" x="160" y="278" fill="#e8eef5">processes</text>
          <text class="qs07__placement qs07__placement--gpus" x="160" y="278" fill="#e8eef5">gpus</text>
          <text class="qs07__placement qs07__placement--jobset" x="160" y="278" fill="#e8eef5">job_set</text>
        </g>
        <g stroke="#4da3ff" stroke-opacity="0.35" stroke-width="1.5">
          <path d="M330 200 C 380 200, 390 79, 440 79"/>
          <path d="M330 200 C 380 200, 390 139, 440 139"/>
          <path d="M330 200 L 440 199"/>
          <path d="M330 200 C 380 200, 390 259, 440 259"/>
          <path d="M330 200 C 380 200, 390 319, 440 319"/>
        </g>
        <text x="440" y="36" fill="#5d7086" font-size="15" font-weight="700" letter-spacing="3">40 JOBS · ONE PER BOOK, DESK, NETTING SET OR GROUP</text>
        <rect class="qs07__cell" style="--l:0.30s; --p:0.30s; --g:0.30s; --j:0.30s" x="440" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.38s; --p:0.33s; --g:0.54s; --j:0.32s" x="500" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.45s; --p:0.36s; --g:0.30s; --j:0.34s" x="560" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.52s; --p:0.39s; --g:0.54s; --j:0.36s" x="620" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.60s; --p:0.42s; --g:0.30s; --j:0.30s" x="680" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.68s; --p:0.45s; --g:0.54s; --j:0.32s" x="740" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.75s; --p:0.48s; --g:0.30s; --j:0.34s" x="800" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.82s; --p:0.51s; --g:0.54s; --j:0.36s" x="860" y="56" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.90s; --p:0.75s; --g:0.78s; --j:0.38s" x="440" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:0.97s; --p:0.78s; --g:1.02s; --j:0.40s" x="500" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.05s; --p:0.81s; --g:0.78s; --j:0.42s" x="560" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.12s; --p:0.84s; --g:1.02s; --j:0.44s" x="620" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.20s; --p:0.87s; --g:0.78s; --j:0.38s" x="680" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.27s; --p:0.90s; --g:1.02s; --j:0.40s" x="740" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.35s; --p:0.93s; --g:0.78s; --j:0.42s" x="800" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.43s; --p:0.96s; --g:1.02s; --j:0.44s" x="860" y="116" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.50s; --p:1.20s; --g:1.26s; --j:0.75s" x="440" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.57s; --p:1.23s; --g:1.50s; --j:0.77s" x="500" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.65s; --p:1.26s; --g:1.26s; --j:0.79s" x="560" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.73s; --p:1.29s; --g:1.50s; --j:0.81s" x="620" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.80s; --p:1.32s; --g:1.26s; --j:0.75s" x="680" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell is-fail" style="--l:1.88s; --p:1.35s; --g:1.50s; --j:0.77s" x="740" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:1.95s; --p:1.38s; --g:1.26s; --j:0.79s" x="800" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.02s; --p:1.41s; --g:1.50s; --j:0.81s" x="860" y="176" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.10s; --p:1.65s; --g:1.74s; --j:0.83s" x="440" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.17s; --p:1.68s; --g:1.98s; --j:0.85s" x="500" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.25s; --p:1.71s; --g:1.74s; --j:0.87s" x="560" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.32s; --p:1.74s; --g:1.98s; --j:0.89s" x="620" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.40s; --p:1.77s; --g:1.74s; --j:0.83s" x="680" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.47s; --p:1.80s; --g:1.98s; --j:0.85s" x="740" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.55s; --p:1.83s; --g:1.74s; --j:0.87s" x="800" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.62s; --p:1.86s; --g:1.98s; --j:0.89s" x="860" y="236" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.70s; --p:2.10s; --g:2.22s; --j:1.20s" x="440" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.77s; --p:2.13s; --g:2.46s; --j:1.22s" x="500" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.85s; --p:2.16s; --g:2.22s; --j:1.24s" x="560" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:2.92s; --p:2.19s; --g:2.46s; --j:1.26s" x="620" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:3.00s; --p:2.22s; --g:2.22s; --j:1.20s" x="680" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:3.07s; --p:2.25s; --g:2.46s; --j:1.22s" x="740" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:3.15s; --p:2.28s; --g:2.22s; --j:1.24s" x="800" y="296" width="46" height="46" rx="8"/>
        <rect class="qs07__cell" style="--l:3.22s; --p:2.31s; --g:2.46s; --j:1.26s" x="860" y="296" width="46" height="46" rx="8"/>
        <g class="qs07__lane qs07__lane--local">
        <path d="M440 356 V362 H906 V356" stroke="#4da3ff" stroke-width="2" stroke-linejoin="round"/>
        <text x="673" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">1 worker · one job at a time</text>
        </g>
        <g class="qs07__lane qs07__lane--procs">
        <rect x="440" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="500" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="560" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="620" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="680" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="740" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="800" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <rect x="860" y="356" width="46" height="5" rx="2.5" fill="#4da3ff"/>
        <text x="673" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">8 CPU worker processes, one per column</text>
        </g>
        <g class="qs07__lane qs07__lane--gpus">
        <rect x="440" y="356" width="106" height="5" rx="2.5" fill="#4da3ff"/><text x="493" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">GPU 0</text>
        <rect x="560" y="356" width="106" height="5" rx="2.5" fill="#4da3ff"/><text x="613" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">GPU 1</text>
        <rect x="680" y="356" width="106" height="5" rx="2.5" fill="#4da3ff"/><text x="733" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">GPU 2</text>
        <rect x="800" y="356" width="106" height="5" rx="2.5" fill="#4da3ff"/><text x="853" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">GPU 3</text>
        </g>
        <g class="qs07__lane qs07__lane--jobset">
        <path d="M440 356 V362 H666 V356" stroke="#4da3ff" stroke-width="2" stroke-linejoin="round"/>
        <path d="M680 356 V362 H906 V356" stroke="#4da3ff" stroke-width="2" stroke-linejoin="round"/>
        <text x="553" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">Machine A · 8 workers</text>
        <text x="793" y="386" fill="#8ea3b8" font-size="14" text-anchor="middle">Machine B · 8 workers</text>
        </g>
        <g class="qs07__late">
          <rect x="960" y="90" width="210" height="220" rx="16" fill="#161b22" stroke="#24502f" stroke-width="1.5"/>
          <text x="988" y="166" fill="#3fb950" font-size="58" font-weight="700" letter-spacing="-1">39/40</text>
          <text x="990" y="200" fill="#e8eef5" font-size="18" font-weight="600">jobs trained</text>
          <circle cx="996" cy="248" r="6" fill="#d29922"/>
          <text x="1012" y="254" fill="#e8eef5" font-size="16" font-weight="600">1 failure, recorded</text>
          <text x="990" y="282" fill="#8ea3b8" font-size="14">same result in every mode</text>
        </g>
      </svg>
      <p class="qs07__hint qs-in" style="--i:4">Click a placement to run it</p>
      <div class="qs07__executors qs-in" style="--i:4">
        <label for="qs07-local" class="qs07__exec qs07__exec--local"><b>Local</b><span>Sequential. The reference result.</span></label>
        <label for="qs07-procs" class="qs07__exec qs07__exec--procs"><b>Processes</b><span>CPU parallel, thread budget per worker</span></label>
        <label for="qs07-gpus" class="qs07__exec qs07__exec--gpus"><b>GPUs</b><span>One worker pinned to each device</span></label>
        <label for="qs07-jobset" class="qs07__exec qs07__exec--jobset"><b>Job set</b><span>Jobs spread across machines, any model</span></label>
      </div>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>07</span></div>
  </div>
</div>
