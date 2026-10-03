# `rade_analytics/assets/`

Anything Dash finds in an app's `assets/` folder is served automatically
at `/_dash-component-suites/...` and loaded into every page. For Rade
Analytics this is where the compiled stylesheet, logo, favicon and
fonts live.

## Files

| File | Served? | Purpose |
| --- | --- | --- |
| `rade.css` | ✅ Yes — Dash auto-loads | Compiled Tailwind bundle. This is the one the browser uses. |
| `tailwind.config.js` | ❌ No — starts with `_` would be cleaner but Dash only ignores hidden files | Canonical source: palette, typography, spacing scale. Reference document. |
| `tailwind.input.css` | ❌ No | Canonical source: `@tailwind` directives + custom `@layer base/components`. Reference document. |
| `logo.svg` / `favicon.ico` | ✅ Yes | Brand assets (arriving in Phase B.2). |

> **Dash serves every static file in this folder,** including
> `tailwind.config.js` and `tailwind.input.css`. That's harmless —
> they're plain text and contain no secrets — but do not put anything
> sensitive here.

## How `rade.css` is produced

Rade deliberately **does not** run a Node / npm toolchain at dev time.
`rade.css` is hand-compiled from `tailwind.input.css` against the config
in `tailwind.config.js`, then committed directly. Dev loop:

1. Need a new utility class (e.g. `grid-cols-8`)?
2. Add it to the right section of `rade.css`, keeping it aligned with
   Tailwind's naming convention.
3. Also add it to `tailwind.input.css` inside the relevant `@layer` if
   it's a new component-level class (`rade-*`), so the reference stays
   in sync.
4. Commit both files in the same change.

### Optional: regenerate via Tailwind CLI

If anyone touching the project has Node available and wants a full
rebuild (handy when the palette moves a lot):

```bash
npx tailwindcss \
    -c src/ui/apps/rade_analytics/assets/tailwind.config.js \
    -i src/ui/apps/rade_analytics/assets/tailwind.input.css \
    -o src/ui/apps/rade_analytics/assets/rade.css \
    --minify
```

This is **never required** for day-to-day development and is not part
of CI. Treat it as a one-off for design-system overhauls.

## Design-spec contract

Everything in this folder must match
[`docs/platform_designs/RADE_UI_DESIGN.md`](../../../../../docs/platform_designs/RADE_UI_DESIGN.md).
If the spec changes, update the config + input + compiled artifact
together. If a component needs a utility that isn't in `rade.css`, add
it; don't work around it with inline styles.
