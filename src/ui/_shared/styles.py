"""
Shared style constants for QuantStrata Dash apps.

Use these dicts with dash.html style= or in inline styles so all apps
look consistent (front-office quant library aesthetic).
"""

# Top-level layout
LAYOUT_STYLES = {
    "container": {
        "fontFamily": "'Segoe UI', system-ui, sans-serif",
        "maxWidth": "1200px",
        "margin": "0 auto",
        "padding": "24px",
    },
    "main": {
        "marginTop": "24px",
    },
}

# Navbar (brand + optional app name)
NAVBAR_STYLES = {
    "navbar": {
        "borderBottom": "1px solid #e0e0e0",
        "padding": "12px 0",
        "marginBottom": "20px",
    },
    "brand": {
        "fontSize": "1.25rem",
        "fontWeight": "600",
        "color": "#1a1a1a",
    },
    "appTitle": {
        "fontSize": "0.95rem",
        "color": "#666",
        "marginLeft": "12px",
    },
}

# Form / input blocks
FORM_STYLES = {
    "formSection": {
        "marginBottom": "24px",
    },
    "inputRow": {
        "display": "grid",
        "gridTemplateColumns": "repeat(auto-fill, minmax(140px, 1fr))",
        "gap": "12px",
        "alignItems": "end",
        "maxWidth": "800px",
    },
    "label": {
        "fontSize": "0.85rem",
        "fontWeight": "500",
        "color": "#444",
        "marginBottom": "4px",
    },
    "input": {
        "padding": "8px 10px",
        "border": "1px solid #ccc",
        "borderRadius": "4px",
        "fontSize": "0.95rem",
    },
    "button": {
        "padding": "10px 20px",
        "backgroundColor": "#2563eb",
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "fontSize": "0.95rem",
        "cursor": "pointer",
        "fontWeight": "500",
    },
    "buttonHover": {"backgroundColor": "#1d4ed8"},
}

# Result / output blocks
RESULT_STYLES = {
    "card": {
        "border": "1px solid #e5e7eb",
        "borderRadius": "8px",
        "padding": "16px",
        "backgroundColor": "#fafafa",
        "marginTop": "16px",
    },
    "pre": {
        "fontFamily": "'Consolas', 'Monaco', monospace",
        "fontSize": "0.9rem",
        "margin": "0",
        "whiteSpace": "pre-wrap",
    },
    "error": {
        "color": "#b91c1c",
        "fontFamily": "'Consolas', monospace",
        "fontSize": "0.9rem",
    },
}

# Footer
FOOTER_STYLES = {
    "footer": {
        "marginTop": "32px",
        "paddingTop": "16px",
        "borderTop": "1px solid #e0e0e0",
        "fontSize": "0.8rem",
        "color": "#666",
    },
}

__all__ = [
    "FOOTER_STYLES",
    "FORM_STYLES",
    "LAYOUT_STYLES",
    "NAVBAR_STYLES",
    "RESULT_STYLES",
]
