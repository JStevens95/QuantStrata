<!--
  QNet deck - 01 The problem.

  Message: before a quant learns anything from a model, they rebuild the
  same plumbing, and that plumbing is where the silent mistakes live.

  The tiles build in one by one; the three red ones pulse once the stack
  is complete. The HTML has no blank lines on purpose: Markdown ends an
  HTML block at the first blank line.
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

  .qs01__stage { display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 1.4cqw; }
  .qs01__tip {
    display: flex; align-items: center; gap: 0.8cqw;
    padding: 1cqw 2.4cqw; border-radius: 999px;
    background: linear-gradient(135deg, #2f86e6, #174f9c);
    box-shadow: 0 0 3cqw rgba(77, 163, 255, 0.35);
    font-size: 1.5cqw; font-weight: 600;
  }
  .qs01__tip i { width: 0.8cqw; height: 0.8cqw; border-radius: 50%; background: #3fb950; }
  .qs01__water {
    width: 100%; padding-top: 0.7cqw; border-top: 1px dashed #3b4a5c;
    text-align: center; font-size: 0.95cqw; letter-spacing: 0.14em; text-transform: uppercase; color: #5d7086;
  }
  .qs01__grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1cqw; width: 100%; }
  .qs01__tile {
    position: relative; min-height: 6cqw; padding: 1.2cqw;
    border: 1px solid #2a3340; border-radius: 0.8cqw; background: #161b22;
    font-size: 1.25cqw; color: #c9d4e0;
  }
  .qs01__tile em { display: block; margin-top: 0.4cqw; font-style: normal; font-size: 0.95cqw; color: #ff7b72; }
  .qs01__tile.is-risk { border-color: #5a2d2d; }
  .qs01__tile.is-risk::after {
    content: ""; position: absolute; top: 0.9cqw; right: 0.9cqw;
    width: 0.7cqw; height: 0.7cqw; border-radius: 50%; background: #f85149;
    animation: qs01-pulse 1.8s ease-out 2s infinite;
  }
  @keyframes qs01-pulse {
    0% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.6); }
    70% { box-shadow: 0 0 0 1cqw rgba(248, 81, 73, 0); }
    100% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }
  }
</style>

<div class="qs qs01">
  <div class="qs__frame">
    <svg class="qs__brand" xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 256 256"
    fill="none"
    aria-hidden="true"
    >
    <defs>
    <radialGradient id="qs01-brand-body" cx="34%" cy="30%" r="72%">
    <stop offset="0%" stop-color="#e6f2ff"/>
    <stop offset="16%" stop-color="#9fcbff"/>
    <stop offset="42%" stop-color="#3f8fe8"/>
    <stop offset="76%" stop-color="#174f9c"/>
    <stop offset="100%" stop-color="#071f3d"/>
    </radialGradient>
    <radialGradient id="qs01-brand-shade" cx="50%" cy="52%" r="50%">
    <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
    <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
    </radialGradient>
    <radialGradient id="qs01-brand-core" cx="40%" cy="35%" r="70%">
    <stop offset="0%" stop-color="#14508f"/>
    <stop offset="100%" stop-color="#0a2c52"/>
    </radialGradient>
    <linearGradient id="qs01-brand-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
    <stop offset="0%" stop-color="#9fd0ff"/>
    <stop offset="55%" stop-color="#ffffff"/>
    <stop offset="100%" stop-color="#7ee08f"/>
    </linearGradient>
    <clipPath id="qs01-brand-clip">
    <circle cx="128" cy="128" r="100"/>
    </clipPath>
    </defs>
    <circle cx="128" cy="128" r="100" fill="url(#qs01-brand-body)"/>
    <g clip-path="url(#qs01-brand-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
    <ellipse cx="128" cy="128" rx="100" ry="34"/>
    <ellipse cx="128" cy="128" rx="36" ry="100"/>
    <ellipse cx="128" cy="96" rx="86" ry="16"/>
    </g>
    <circle cx="128" cy="128" r="100" fill="url(#qs01-brand-shade)"/>
    <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
    <g id="qs01-brand-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
    <path id="qs01-brand-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qs01-brand-signal)" stroke-width="2.4"/>
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
    <circle cx="128" cy="124" r="37" fill="url(#qs01-brand-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
    <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
    <circle cx="128" cy="122" r="18"/>
    <path d="M134 128 L148 142"/>
    </g>
    </svg>
    <p class="qs__kicker qs-in">01 · The problem</p>
    <h1 class="qs__title qs-in" style="--i:1">Every new model rebuilds the same plumbing</h1>
    <p class="qs__sub qs-in" style="--i:2">And that plumbing is where the silent mistakes live.</p>
    <div class="qs__stage qs01__stage">
      <div class="qs01__tip qs-in" style="--i:3"><i></i>Your model idea</div>
      <div class="qs01__water qs-in" style="--i:4">Rebuilt for every model, before anything is learned</div>
      <div class="qs01__grid">
        <div class="qs01__tile qs-in" style="--i:5">Data loader</div>
        <div class="qs01__tile is-risk qs-in" style="--i:6">Train / test split<em>Shuffled: look-ahead</em></div>
        <div class="qs01__tile qs-in" style="--i:7">Training loop</div>
        <div class="qs01__tile is-risk qs-in" style="--i:8">Scaling<em>Fitted on test data</em></div>
        <div class="qs01__tile qs-in" style="--i:9">Early stopping</div>
        <div class="qs01__tile qs-in" style="--i:10">Metrics and plots</div>
        <div class="qs01__tile is-risk qs-in" style="--i:11">Save and reload<em>Breaks after a refactor</em></div>
        <div class="qs01__tile qs-in" style="--i:12">Run across 40 books</div>
        <div class="qs01__tile qs-in" style="--i:13">GPU set-up</div>
        <div class="qs01__tile qs-in" style="--i:14">Audit trail</div>
      </div>
    </div>
    <div class="qs__foot"><span>RADE ANALYTICS · INTERNAL</span><span>01</span></div>
  </div>
</div>
