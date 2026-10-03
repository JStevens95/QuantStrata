"""Rade wordmark + logo — used by sidebar, splash hero and topbar.

A single canonical renderer so the wordmark, logo size ratio and
gradient treatment never drift across pages.  Four named sizes cover
every in-app usage; pages don't supply raw Tailwind classes.

Design spec anchors
-------------------
* §3 Typography — Inter, ``tracking-[-0.02em]`` on the wordmark.
* §6 Components — ``rade-brand-gradient-text`` from ``rade.css``.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from dash import html
from dash.development.base_component import Component


BrandSize = Literal["sm", "md", "lg", "xl"]


# Mark side, wordmark size + gap are tuned so the logo and text baseline
# align at every size. Keep this table exhaustive so linters flag any
# missing size at change-time rather than at render-time.
# NOTE: keep wordmark sizes in the set {sm, base, lg, 2xl, 4xl} —
# ``text-5xl`` is outside the content-scan that produced the shipped
# ``rade.css`` so it would render as an inherited (regular) size.
_SIZE_TABLE: Dict[BrandSize, Dict[str, str]] = {
    "sm": {"mark": "w-6 h-6",   "word": "text-sm",  "gap": "gap-2"},
    "md": {"mark": "w-8 h-8",   "word": "text-lg",  "gap": "gap-3"},
    "lg": {"mark": "w-12 h-12", "word": "text-2xl", "gap": "gap-3"},
    "xl": {"mark": "w-20 h-20", "word": "text-4xl", "gap": "gap-5"},
}


def Brand(
    *,
    size: BrandSize = "md",
    show_wordmark: bool = True,
    href: Optional[str] = None,
    brand_id: Optional[str] = None,
    className: str = "",
) -> Component:
    """Render the Rade mark and (optionally) the wordmark next to it.

    Parameters
    ----------
    size
        One of ``"sm" | "md" | "lg" | "xl"``.  Controls both the logo
        dimensions and the wordmark font-size so the pair stays
        balanced.
    show_wordmark
        Set ``False`` for tight spots where only the mark fits.
    href
        If supplied, renders as an anchor that navigates to this URL.
        Used in the sidebar header to send users back to ``/``.
    brand_id
        Optional DOM id for tests / callbacks.  Named ``brand_id`` (not
        ``id``) to avoid shadowing the Python built-in — every primitive
        in this package follows the same ``{component}_id`` convention.
    className
        Extra Tailwind classes appended to the wrapper (usually none;
        layout belongs to the caller's container).
    """
    scale = _SIZE_TABLE[size]

    body = [
        html.Img(
            src="/assets/logo.svg",
            className=scale["mark"],
            alt="Rade logo",
        )
    ]
    if show_wordmark:
        body.append(
            html.Span(
                "Rade",
                className=(
                    f"{scale['word']} font-semibold "
                    "tracking-tight rade-brand-gradient-text"
                ),
            )
        )

    wrapper_class = (
        f"flex items-center {scale['gap']} {className}"
    ).strip()

    # Dash rejects ``id=None``; only include the kwarg when set.
    id_kwargs = {"id": brand_id} if brand_id is not None else {}

    if href is not None:
        # ``no-underline`` isn't in the shipped rade.css so we strip
        # the default anchor underline via inline style instead.
        return html.A(
            href=href,
            className=wrapper_class,
            style={"textDecoration": "none"},
            children=body,
            **id_kwargs,
        )
    return html.Div(
        className=wrapper_class,
        children=body,
        **id_kwargs,
    )


__all__ = ["Brand", "BrandSize"]
