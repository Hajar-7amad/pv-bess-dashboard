"""Custom Plot Builder – lightweight scientific plotting without code."""
from __future__ import annotations

import streamlit as st

from modules import state, ui
from modules.explorer import color_options, data_controls
from modules.export import csv_bytes, fig_image_bytes
from modules.plotting import build_chart

ui.header("Custom Plot Builder", "Build publication-style figures from any combination of parameters, "
          "batteries, periods and filters.")
ui.require_data()

with st.container(border=True):
    st.markdown("**1 · Data**")
    df, ch = data_controls("pb", show_soc=True)

vars_ = ui.numeric_vars(df)
with st.container(border=True):
    st.markdown("**2 · Chart**")
    c1, c2, c3, c4 = st.columns(4)
    chart = c1.selectbox("Chart type", ["Line chart", "Scatter plot", "Bar chart", "Histogram", "Box plot",
                                        "Heatmap", "Correlation matrix"], key="pb_chart")
    xopts = ["timestamp"] + vars_
    x = c2.selectbox("X parameter", xopts, format_func=lambda c: "Time" if c == "timestamp" else ui.var_label(c),
                     key="pb_x")
    y = c3.selectbox("Y parameter", vars_, format_func=ui.var_label, key="pb_y")
    more = c4.multiselect("Additional Y-series", [v for v in vars_ if v != y], format_func=ui.var_label, key="pb_more")
    c5, c6, c7, c8 = st.columns(4)
    y2 = c5.selectbox("Second Y-axis (line, time axis)", ["(none)"] + vars_,
                      format_func=lambda c: c if c == "(none)" else ui.var_label(c), key="pb_y2")
    color = c6.selectbox("Colour variable", color_options(df), key="pb_color",
                         format_func=lambda c: c if c in ("(none)", "battery", "operating_mode") else ui.var_label(c))
    ts_mode = c7.selectbox("Multi-series layout (time axis)", ["shared", "panels"], key="pb_tsmode",
                           format_func={"shared": "One axis", "panels": "Stacked panels"}.get)
    nbins = c8.number_input("Histogram bins", 5, 300, 50, key="pb_bins")

with st.container(border=True):
    st.markdown("**3 · Appearance**")
    c1, c2, c3 = st.columns([2, 1.5, 1.5])
    title = c1.text_input("Plot title", f"{ui.var_label(y)}", key="pb_title")
    xlabel = c2.text_input("X-axis label (blank = automatic)", "", key="pb_xl")
    ylabel = c3.text_input("Y-axis label (blank = automatic)", "", key="pb_yl")
    c4, c5, c6, c7 = st.columns(4)
    logx = c4.checkbox("Log X", key="pb_lx")
    logy = c5.checkbox("Log Y", key="pb_ly")
    legend = c6.checkbox("Show legend", True, key="pb_leg")
    height = c7.slider("Height [px]", 300, 1000, 500, 20, key="pb_h")

fig = build_chart(df, chart, x, [y] + more, y2=None if y2 == "(none)" else y2,
                  color=None if color == "(none)" else color, title=title, xlabel=xlabel, ylabel=ylabel,
                  log_x=logx, log_y=logy, legend=legend, ts_mode=ts_mode, colors=state.colors(),
                  nbins=int(nbins), height=height)
ui.show_fig(fig, "pb_fig", title.replace(" ", "_")[:40] or "custom_plot")

c1, c2, c3 = st.columns(3)
fmt = state.settings().get("figure_export_format", "png")
if c1.button(f"Render static {fmt.upper()} (server-side)", key="pb_static"):
    img = fig_image_bytes(fig, fmt, state.settings().get("figure_scale", 2))
    if img:
        st.session_state["pb_img"] = (img, fmt)
    else:
        st.warning("Server-side export needs Kaleido with a Chromium browser. Use the camera icon in the chart "
                   "toolbar instead (format follows Settings).")
if "pb_img" in st.session_state:
    img, f = st.session_state["pb_img"]
    c2.download_button(f"Download {f.upper()}", img, file_name=f"custom_plot.{f}", key="pb_img_dl")
c3.download_button("Download plotted data (CSV)", csv_bytes(df), file_name="custom_plot_data.csv", key="pb_csv")
