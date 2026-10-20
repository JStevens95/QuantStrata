<!--
  QNet deck - 02 What QNet is.

  Message: a model author writes three small files; QNet runs the whole
  lifecycle around them, whatever the library.

  Dots run continuously from the three files into the mark, and from the
  mark out to the four pipelines. The HTML has no blank lines on purpose:
  Markdown ends an HTML block at the first blank line.
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

  .qs02__stage { display: flex; flex-direction: column; gap: 1.6cqw; }
  .qs02__diagram { width: 100%; flex: 1; min-height: 0; }
  .qs02__diagram text { font-family: Inter, "Segoe UI", Helvetica, Arial, sans-serif; }
  .qs02__engines { display: flex; align-items: center; justify-content: center; gap: 1cqw; font-size: 1.15cqw; color: #8ea3b8; }
  .qs02__engines b {
    padding: 0.35cqw 1.1cqw; border: 1px solid #2d5680; border-radius: 999px;
    font-weight: 600; color: #cfe5ff; background: rgba(77, 163, 255, 0.08);
  }
</style>

<div class="qs qs02">
  <div class="qs__frame">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs02-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs02-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs02-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs02-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs02-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs02-brand-body)"/>
    <g clip-path="url(#qs02-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs02-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs02-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
    <path id="qs02-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs02-brand-signal)" stroke-width="2.4"/>
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
    <circle cx="128" cy="124" r="37" fill="url(#qs02-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">02 · What is QNet</p>
    <h1 class="qs__title qs-in" style="--i:1">You bring the maths. QNet runs everything else.</h1>
    <div class="qs__stage qs02__stage">
      <svg class="qs02__diagram qs-in" style="--i:2" viewBox="0 0 1200 420" fill="none" role="img" aria-label="Three model files flow into QNet, which runs train, evaluate, infer and tune">
        <text x="40" y="22" fill="#5d7086" font-size="15" font-weight="700" letter-spacing="3">YOU WRITE</text>
        <text x="1160" y="22" fill="#5d7086" font-size="15" font-weight="700" letter-spacing="3" text-anchor="end">QNET RUNS</text>
        <g stroke="#4da3ff" stroke-opacity="0.45" stroke-width="2">
          <path id="qs02-in-1" d="M340 85 C 420 85, 410 210, 478 210"/>
          <path id="qs02-in-2" d="M340 210 L 478 210"/>
          <path id="qs02-in-3" d="M340 335 C 420 335, 410 210, 478 210"/>
        </g>
        <g stroke="#3fb950" stroke-opacity="0.45" stroke-width="2">
          <path id="qs02-out-1" d="M722 210 C 790 210, 790 75, 860 75"/>
          <path id="qs02-out-2" d="M722 210 C 790 210, 790 165, 860 165"/>
          <path id="qs02-out-3" d="M722 210 C 790 210, 790 255, 860 255"/>
          <path id="qs02-out-4" d="M722 210 C 790 210, 790 345, 860 345"/>
        </g>
        <g fill="#161b22" stroke="#2a3340" stroke-width="1.5">
          <rect x="40" y="40" width="300" height="90" rx="14"/>
          <rect x="40" y="165" width="300" height="90" rx="14"/>
          <rect x="40" y="290" width="300" height="90" rx="14"/>
        </g>
        <g font-size="22" font-weight="600" fill="#e8eef5" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">
          <text x="68" y="80">model.py</text>
          <text x="68" y="205">data.py</text>
          <text x="68" y="330">state.py</text>
        </g>
        <g font-size="16" fill="#8ea3b8">
          <text x="68" y="108">the architecture</text>
          <text x="68" y="233">where its data comes from</text>
          <text x="68" y="358">what it fits, for inference</text>
        </g>
        <circle cx="600" cy="210" r="150" fill="#4da3ff" fill-opacity="0.07"/>
        <svg x="472" y="82" width="256" height="256" viewBox="0 0 256 256">
          <defs>
          <radialGradient id="qs02-mark-body" cx="34%" cy="30%" r="72%">
          <stop offset="0%" stop-color="#e6f2ff"/>
          <stop offset="16%" stop-color="#9fcbff"/>
          <stop offset="42%" stop-color="#3f8fe8"/>
          <stop offset="76%" stop-color="#174f9c"/>
          <stop offset="100%" stop-color="#071f3d"/>
          </radialGradient>
          <radialGradient id="qs02-mark-shade" cx="50%" cy="52%" r="50%">
          <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
          <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
          </radialGradient>
          <radialGradient id="qs02-mark-core" cx="40%" cy="35%" r="70%">
          <stop offset="0%" stop-color="#14508f"/>
          <stop offset="100%" stop-color="#0a2c52"/>
          </radialGradient>
          <linearGradient id="qs02-mark-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#9fd0ff"/>
          <stop offset="55%" stop-color="#ffffff"/>
          <stop offset="100%" stop-color="#7ee08f"/>
          </linearGradient>
          <clipPath id="qs02-mark-clip">
          <circle cx="128" cy="128" r="100"/>
          </clipPath>
          </defs>
          <circle cx="128" cy="128" r="100" fill="url(#qs02-mark-body)"/>
          <g clip-path="url(#qs02-mark-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
          <ellipse cx="128" cy="128" rx="100" ry="34"/>
          <ellipse cx="128" cy="128" rx="36" ry="100"/>
          <ellipse cx="128" cy="96" rx="86" ry="16"/>
          </g>
          <circle cx="128" cy="128" r="100" fill="url(#qs02-mark-shade)"/>
          <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
          <g id="qs02-mark-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
          <path id="qs02-mark-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs02-mark-signal)" stroke-width="2.4"/>
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
          <circle cx="128" cy="124" r="37" fill="url(#qs02-mark-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
          <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
          <circle cx="128" cy="122" r="18"/>
          <path d="M134 128 L148 142"/>
          </g>
        </svg>
        <g fill="#161b22" stroke="#24502f" stroke-width="1.5">
          <rect x="860" y="40" width="300" height="70" rx="35"/>
          <rect x="860" y="130" width="300" height="70" rx="35"/>
          <rect x="860" y="220" width="300" height="70" rx="35"/>
          <rect x="860" y="310" width="300" height="70" rx="35"/>
        </g>
        <g fill="#3fb950">
          <circle cx="892" cy="75" r="6"/>
          <circle cx="892" cy="165" r="6"/>
          <circle cx="892" cy="255" r="6"/>
          <circle cx="892" cy="345" r="6"/>
        </g>
        <g font-size="22" font-weight="600" fill="#e8eef5">
          <text x="914" y="83">Train</text>
          <text x="914" y="173">Evaluate</text>
          <text x="914" y="263">Infer</text>
          <text x="914" y="353">Tune</text>
        </g>
        <g font-size="15" fill="#8ea3b8" text-anchor="end">
          <text x="1136" y="81">fit · stop · checkpoint</text>
          <text x="1136" y="171">in original units</text>
          <text x="1136" y="261">serve predictions</text>
          <text x="1136" y="351">data built once</text>
        </g>
        <g fill="#9fd0ff">
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="0s"><mpath href="#qs02-in-1"/></animateMotion></circle>
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-1.6s"><mpath href="#qs02-in-2"/></animateMotion></circle>
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-0.8s"><mpath href="#qs02-in-3"/></animateMotion></circle>
        </g>
        <g fill="#7ee08f">
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-2s"><mpath href="#qs02-out-1"/></animateMotion></circle>
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-1.4s"><mpath href="#qs02-out-2"/></animateMotion></circle>
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-0.8s"><mpath href="#qs02-out-3"/></animateMotion></circle>
          <circle r="5"><animateMotion dur="2.4s" repeatCount="indefinite" begin="-0.2s"><mpath href="#qs02-out-4"/></animateMotion></circle>
        </g>
      </svg>
      <p class="qs02__engines qs-in" style="--i:4">Any model, one lifecycle, on <b>PyTorch</b><b>JAX</b><b>XGBoost</b><b>LightGBM</b><b>scikit-learn</b></p>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>02</span></div>
  </div>
</div>
