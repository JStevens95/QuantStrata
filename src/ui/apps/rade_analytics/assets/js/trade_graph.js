/* ──────────────────────────────────────────────────────────────────
 * Trade-Graph sub-tab — clientside callbacks.
 *
 * Two actions live here, both targeting the Cytoscape instance the
 * Dash app mounts at id="eval-trade-graph-cytoscape":
 *
 *   1. fit_view   — re-runs cy.fit() so the graph centres + zooms
 *                   to fill the viewport.  Useful after the user
 *                   has panned / zoomed away.
 *   2. export_png — calls cy.png({...}) and triggers a download
 *                   via a synthetic <a> element.
 *
 * Both functions are referenced from Python via:
 *
 *     ClientsideFunction(namespace="trade_graph",
 *                        function_name="fit_view")
 *
 * Page Contract reference: §6 Lever P2 (clientside_callback for
 * trivial UI / non-stateful actions).
 *
 * Why we hunt for the cy instance via dash_cytoscape's internal
 * registry: dash_cytoscape doesn't expose a clean "give me the
 * cy instance for id=X" hook.  The component stashes the cy
 * reference on the underlying React component, which we reach via
 * ``window.cy`` (set by some integrations) or by reading the
 * canvas DOM and walking the React fiber tree as a last resort.
 *
 * The lookup intentionally guards against "Cytoscape not ready
 * yet" — when the user clicks Fit before the graph has finished
 * its first render, we silently no-op rather than throwing.
 * ────────────────────────────────────────────────────────────────── */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.trade_graph = (function () {
    "use strict";

    /* Locate the live Cytoscape instance for the given DOM id.
     *
     * dash_cytoscape attaches the cy instance as a property on the
     * outer container element; we read it back via the cy global
     * pattern. */
    function getCy(cytoscapeId) {
        const el = document.getElementById(cytoscapeId);
        if (!el) {
            return null;
        }
        /* dash_cytoscape (v0.3+) exposes the cy instance as a
         * property on the wrapping div.  Older versions stashed it
         * on the canvas element — we check both. */
        if (el._cyreg && el._cyreg.cy) {
            return el._cyreg.cy;
        }
        if (el.__cy) {
            return el.__cy;
        }
        /* Last resort — walk the children for a canvas with a cy
         * back-reference (older dash_cytoscape builds). */
        const canvas = el.querySelector("canvas");
        if (canvas && canvas._cy) {
            return canvas._cy;
        }
        return null;
    }

    /* Trigger a download of a base64 PNG payload as a file. */
    function triggerDownload(dataUrl, fileName) {
        const a = document.createElement("a");
        a.href = dataUrl;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    /* Fit-view — re-runs the layout's fit/center logic.
     *
     * Returns dash.no_update so we don't churn the button's n_clicks
     * (which is the Dash callback's Output sink). */
    function fit_view(_n_clicks, cytoscapeId) {
        if (!_n_clicks) {
            return window.dash_clientside.no_update;
        }
        const cy = getCy(cytoscapeId);
        if (!cy) {
            return window.dash_clientside.no_update;
        }
        cy.fit(undefined, 30);    // 30 px padding — matches layout default
        cy.center();
        return window.dash_clientside.no_update;
    }

    /* Export-PNG — generates a high-res PNG of the current graph
     * state and triggers a browser download. */
    function export_png(_n_clicks, cytoscapeId) {
        if (!_n_clicks) {
            return window.dash_clientside.no_update;
        }
        const cy = getCy(cytoscapeId);
        if (!cy) {
            return window.dash_clientside.no_update;
        }

        /* 2× scale gives us a crisp image for retina displays
         * without bloating the file size. */
        const dataUrl = cy.png({
            output: "base64uri",
            full:   true,
            scale:  2,
            bg:     "#0f172a",     // slate-900 — matches dark theme
        });

        const ts = new Date().toISOString().replace(/[:.]/g, "-");
        triggerDownload(dataUrl, `trade-graph-${ts}.png`);

        return window.dash_clientside.no_update;
    }

    return {
        fit_view:   fit_view,
        export_png: export_png,
    };
})();
