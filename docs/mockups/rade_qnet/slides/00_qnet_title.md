<!--
  QNet deck - 00 Title.

  The title slide from logos/platform.html. The lockup is inlined from
  logos/qnet-lockup-hero.svg so the slide renders wherever it is copied.
  The HTML has no blank lines on purpose: Markdown ends an HTML block at
  the first blank line.
-->

<style>
  .qnet-title {
    position: relative;
    aspect-ratio: 16 / 9;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #2a3340;
    background:
      radial-gradient(60% 80% at 22% 50%, rgba(77, 163, 255, 0.18), transparent 70%),
      radial-gradient(40% 60% at 85% 85%, rgba(63, 185, 80, 0.08), transparent 70%),
      #0d1117;
    display: grid;
    align-content: center;
    padding: 0 7%;
    box-sizing: border-box;
    color: #e8eef5;
    font: 15px/1.5 Inter, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  .qnet-title__logo { width: 78%; height: auto; display: block; }
  .qnet-title__line {
    margin: 28px 0 0 2%;
    font-size: clamp(16px, 2vw, 24px);
    color: #8ea3b8;
  }
  .qnet-title__line b { color: #e8eef5; font-weight: 600; }
  .qnet-title__foot {
    position: absolute;
    left: 7%;
    right: 7%;
    bottom: 6%;
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #5d7086;
    letter-spacing: 0.08em;
  }
</style>

<div class="qnet-title">
    <svg class="qnet-title__logo" xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 820 256"
         fill="none"
         role="img"
         aria-label="Rade QNet, Quantitative Neural Engine and Toolkit">
      <title>Rade QNet</title>
      <defs>
        <radialGradient id="qnet-hero-body" cx="34%" cy="30%" r="72%">
          <stop offset="0%" stop-color="#e6f2ff"/>
          <stop offset="16%" stop-color="#9fcbff"/>
          <stop offset="42%" stop-color="#3f8fe8"/>
          <stop offset="76%" stop-color="#174f9c"/>
          <stop offset="100%" stop-color="#071f3d"/>
        </radialGradient>
        <radialGradient id="qnet-hero-shade" cx="50%" cy="52%" r="50%">
          <stop offset="70%" stop-color="#071f3d" stop-opacity="0"/>
          <stop offset="100%" stop-color="#030f1f" stop-opacity="0.6"/>
        </radialGradient>
        <radialGradient id="qnet-hero-core" cx="40%" cy="35%" r="70%">
          <stop offset="0%" stop-color="#14508f"/>
          <stop offset="100%" stop-color="#0a2c52"/>
        </radialGradient>
        <linearGradient id="qnet-hero-signal" x1="16" y1="0" x2="240" y2="0" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stop-color="#9fd0ff"/>
          <stop offset="55%" stop-color="#ffffff"/>
          <stop offset="100%" stop-color="#7ee08f"/>
        </linearGradient>
        <clipPath id="qnet-hero-clip">
          <circle cx="128" cy="128" r="100"/>
        </clipPath>
      </defs>
      <circle cx="128" cy="128" r="100" fill="url(#qnet-hero-body)"/>
      <g clip-path="url(#qnet-hero-clip)" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.1">
        <ellipse cx="128" cy="128" rx="100" ry="34"/>
        <ellipse cx="128" cy="128" rx="36" ry="100"/>
        <ellipse cx="128" cy="96" rx="86" ry="16"/>
      </g>
      <circle cx="128" cy="128" r="100" fill="url(#qnet-hero-shade)"/>
      <ellipse cx="98" cy="84" rx="40" ry="22" fill="#ffffff" opacity="0.14" transform="rotate(-28 98 84)"/>
      <g id="qnet-hero-links" stroke="#ffffff" stroke-width="1.1" stroke-opacity="0.34" stroke-linecap="round">
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
        <path id="qnet-hero-route" d="M16 124 L68 144 L128 104 L188 104 L240 124" stroke="url(#qnet-hero-signal)" stroke-width="2.4"/>
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
      <circle cx="128" cy="124" r="37" fill="url(#qnet-hero-core)" stroke="#9fd0ff" stroke-opacity="0.55" stroke-width="1.5"/>
      <g stroke="#ffffff" stroke-width="8" stroke-linecap="round">
        <circle cx="128" cy="122" r="18"/>
        <path d="M134 128 L148 142"/>
      </g>
      <text x="280" y="70"
            fill="#4da3ff"
            font-family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
            font-size="15"
            font-weight="700"
            letter-spacing="5">RADE</text>
      <text x="274" y="150"
            fill="#f4f7fb"
            font-family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
            font-size="92"
            font-weight="700"
            letter-spacing="-2">QNet</text>
      <text x="280" y="192"
            fill="#8ea3b8"
            font-family="Inter, Segoe UI, Helvetica Neue, Arial, sans-serif"
            font-size="15"
            font-weight="600"
            letter-spacing="2.4">QUANTITATIVE NEURAL ENGINE &amp; TOOLKIT</text>
    </svg>
  <p class="qnet-title__line"><b>From data to decisions.</b> One platform for quantitative models, from source to signal.</p>
  <div class="qnet-title__foot"><span>RADE ANALYTICS · INTERNAL</span><span>2026</span></div>
</div>
