"""
Export helpers: CSV tables, figures (PNG/SVG/HTML) and a zipped results bundle.

PNG/SVG of interactive charts can always be downloaded from the chart toolbar (camera
icon; format set in Settings). Server-side static export uses Kaleido when it and a
Chromium browser are available; otherwise the HTML export (fully interactive, offline
viewable) is offered.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go


def csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def fig_html_bytes(fig: go.Figure, include_plotlyjs="cdn") -> bytes:
    return fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True,
                       config={"displaylogo": False}).encode("utf-8")


def fig_image_bytes(fig: go.Figure, fmt: str = "png", scale: int = 2,
                    width: int = 1200, height: Optional[int] = None) -> Optional[bytes]:
    """Static image via Kaleido. Returns None if Kaleido/Chromium is unavailable."""
    try:
        return fig.to_image(format=fmt, scale=scale, width=width,
                            height=height or fig.layout.height or 500)
    except Exception:  # kaleido missing, no browser, etc.
        return None


def mpl_bytes(fig, fmt: str = "png", dpi: int = 200) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def zip_bundle(tables: Dict[str, pd.DataFrame], texts: Optional[Dict[str, str]] = None,
               binaries: Optional[Dict[str, bytes]] = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is not None and not df.empty:
                z.writestr(f"{name}.csv", csv_bytes(df))
        for name, txt in (texts or {}).items():
            z.writestr(name, txt)
        for name, data in (binaries or {}).items():
            if data:
                z.writestr(name, data)
    return buf.getvalue()


def settings_json(settings: dict, batteries: list) -> str:
    return json.dumps({"settings": settings, "batteries": batteries}, indent=2, default=str)
