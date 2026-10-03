/* ──────────────────────────────────────────────────────────────────
 * Evaluation page — clientside callbacks.
 *
 * Runs entirely in the browser; no server round-trips.  Functions are
 * referenced from Python via:
 *
 *     ClientsideFunction(namespace="evaluation",
 *                        function_name="update_filter_ui")
 *
 * Page Contract reference: §6 Lever P2 (clientside_callback for trivial
 * UI), §3 Rule L1 (initial render still happens server-side via
 * render_filter_chips so the page is usable before any callback fires).
 *
 * The DOM trees produced here MUST match the structure produced by
 * components.evaluation_filter_bar.render_filter_chips on the Python
 * side, otherwise the initial render and the post-mount updates will
 * diverge visually.  Both paths render the chip as
 *
 *     <div class="rade-filter-chip">
 *       <span class="rade-filter-chip-label">{label}: {value}</span>
 *       <button id={type:"eval-filter-chip-close",dimension:dim}
 *               class="rade-filter-chip-close"
 *               aria-label="Remove {dim} filter">×</button>
 *     </div>
 *
 * All visual styling lives in rade.css under `.rade-filter-chip*`.
 * ────────────────────────────────────────────────────────────────── */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.evaluation = (function () {
    "use strict";

    /* Human-readable label per dimension; mirrors _CHIP_LABELS in
     * components/evaluation_filter_bar.py.  Keep the two in sync. */
    const CHIP_LABELS = {
        asset_class: "Asset",
        currency:    "CCY",
        desk:        "Desk",
        product:     "Product",
        date:        "Date",
    };

    /* Build a Dash component dict for one chip.  Using the html.* layer
     * (dash_html_components) instead of dmc.Badge keeps us free of
     * Mantine-internal serialisation; CSS does the visual work. */
    function makeChip(dimension, text) {
        return {
            namespace: "dash_html_components",
            type: "Div",
            props: {
                className: "rade-filter-chip",
                children: [
                    {
                        namespace: "dash_html_components",
                        type: "Span",
                        props: {
                            className: "rade-filter-chip-label",
                            children: text,
                        },
                    },
                    {
                        namespace: "dash_html_components",
                        type: "Button",
                        props: {
                            id: {
                                type: "eval-filter-chip-close",
                                dimension: dimension,
                            },
                            className: "rade-filter-chip-close",
                            children: "\u00D7",  // multiplication-sign ×
                            "aria-label": "Remove " + dimension + " filter",
                        },
                    },
                ],
            },
        };
    }

    /* Compose the value-text for a multi-value dimension chip — first
     * three values verbatim, "+N" overflow indicator after that. */
    function formatValueList(values) {
        if (!Array.isArray(values) || values.length === 0) return "";
        const head = values.slice(0, 3).join(", ");
        const overflow = values.length > 3 ? " +" + (values.length - 3) : "";
        return head + overflow;
    }

    /* Compose the "from – to" text for the date range chip.  Mirrors
     * the Python helper in render_filter_chips. */
    function formatDateRange(dateRange) {
        if (!Array.isArray(dateRange)) return null;
        const from = dateRange[0] || null;
        const to   = dateRange[1] || null;
        if (!from && !to) return null;
        if (from && to)   return from + "\u2013" + to;  // en-dash –
        return from || to;
    }

    /* Public entry point.  Inputs are the raw values from the five
     * filter controls (four MultiSelects + DatePickerInput).  Returns
     * a 3-tuple matching the Output ordering in evaluation_cb.py:
     *
     *     [chips, toggleLabel, clearAllStyle]
     *
     * - chips: list of chip component dicts (or [] for none).
     * - toggleLabel: "{n} active" or "" when no filters are set.
     * - clearAllStyle: {} (visible) when ≥1 filter, {display: "none"}
     *                  when none.  Inline style is the simplest
     *                  visibility lever; CSS could replace this later. */
    function update_filter_ui(asset_class, currency, desk, product, dateRange) {
        const dimensions = [
            ["asset_class", asset_class],
            ["currency",    currency],
            ["desk",        desk],
            ["product",     product],
        ];

        const chips = [];
        let activeCount = 0;

        for (const [dim, values] of dimensions) {
            if (Array.isArray(values) && values.length > 0) {
                activeCount += 1;
                const text = CHIP_LABELS[dim] + ": " + formatValueList(values);
                chips.push(makeChip(dim, text));
            }
        }

        const dateText = formatDateRange(dateRange);
        if (dateText) {
            activeCount += 1;
            chips.push(makeChip("date", "Date: " + dateText));
        }

        const toggleLabel  = activeCount > 0 ? activeCount + " active" : "";
        const clearAllStyle = activeCount > 0 ? {} : { display: "none" };

        return [chips, toggleLabel, clearAllStyle];
    }

    return {
        update_filter_ui: update_filter_ui,
    };
})();
