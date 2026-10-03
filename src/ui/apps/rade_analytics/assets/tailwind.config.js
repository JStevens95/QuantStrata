/**
 * Tailwind config — REFERENCE ONLY.
 *
 * Rade Analytics ships a pre-compiled `rade.css` in this folder; Dash
 * serves that file directly.  This config documents the _source of
 * truth_ for the palette, typography and spacing scale so anyone
 * auditing the stylesheet can see how it was authored, and so it can
 * be plugged straight back into the Tailwind CLI if we ever decide to
 * re-enable the build step.
 *
 * Rebuild (only if Node is available on the environment doing the
 * rebuild — not required for day-to-day dev):
 *
 *     npx tailwindcss \
 *         -c src/ui/apps/rade_analytics/assets/tailwind.config.js \
 *         -i src/ui/apps/rade_analytics/assets/tailwind.input.css \
 *         -o src/ui/apps/rade_analytics/assets/rade.css \
 *         --minify
 *
 * Palette, scale and typography mirror RADE_UI_DESIGN.md §§2–4.  If
 * the spec changes, update this file AND regenerate rade.css.
 */

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    // Every Python file that might emit a className string.
    "../../../**/*.py",
  ],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "IBM Plex Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      letterSpacing: {
        "brand-tight": "-0.02em",
      },
      borderRadius: {
        // Tailwind defaults already cover sm/md/lg/xl/2xl/3xl — noted
        // here so §4 of the design spec can point at a single file.
      },
      ringColor: {
        brand: "#8b5cf6", // violet-500 — default focus ring per §10
      },
      backgroundImage: {
        "brand-gradient":
          "linear-gradient(to right, #8b5cf6, #22d3ee)", // violet-500 → cyan-400
      },
    },
  },
  plugins: [],
};
