"""
PV-BESS Explainable Analytics & Battery Performance Platform
============================================================

Entry point:   streamlit run app.py

An analytical layer above the EPower Monitor EMS for monitoring, battery charging
analysis (capacity-normalized charging rate R_cn = P_charge / E_nominal), battery
comparison, XGBoost prediction and SHAP explainability.

Read-only by design: the platform never issues battery control commands.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from config import APP_SHORT, APP_TITLE, APP_VERSION, SYSTEM_INFO  # noqa: E402
from modules import state, ui  # noqa: E402

st.set_page_config(page_title=APP_SHORT, page_icon="🔋", layout="wide",
                   initial_sidebar_state="expanded")
state.init_state()
ui.inject_css()

# Saudi Water Authority logo
logo_path = ROOT / "assets" / "logo.png"

if logo_path.exists():
    st.logo(str(logo_path), size="large")

st.markdown("""
<style>

[data-testid="stSidebarHeader"] {
    height: 165px !important;
    min-height: 165px !important;
    padding: 15px 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: visible !important;
}

[data-testid="stSidebarHeader"] img {
    width: 200px !important;
    height: auto !important;
    max-height: 140px !important;
    object-fit: contain !important;
}

[data-testid="stSidebarHeader"] > div {
    overflow: visible !important;
}

</style>
""", unsafe_allow_html=True)

P = "pages/"
pages = {
    "Data": [
        st.Page(P + "data_import.py", title="Data Import & Mapping", icon=":material/upload_file:"),
        st.Page(P + "overview.py", title="System Overview", icon=":material/dashboard:", default=True),
        st.Page(P + "data_explorer.py", title="Data Explorer", icon=":material/query_stats:"),
        st.Page(P + "plot_builder.py", title="Custom Plot Builder", icon=":material/insert_chart:"),
        st.Page(P + "data_quality.py", title="Data Quality", icon=":material/fact_check:"),
    ],
    "Battery analytics": [
        st.Page(P + "battery_analysis.py", title="Battery Analysis", icon=":material/battery_charging_full:"),
        st.Page(P + "charging_analysis.py", title="Charging Analysis (R_cn)", icon=":material/bolt:"),
        st.Page(P + "battery_comparison.py", title="Battery Performance Comparison", icon=":material/compare_arrows:"),
    ],
    "Explainable ML": [
        st.Page(P + "machine_learning.py", title="Machine Learning", icon=":material/model_training:"),
        st.Page(P + "shap_dashboard.py", title="SHAP Explainability", icon=":material/psychology:"),
    ],
    "Output & system": [
        st.Page(P + "reports.py", title="Report / Export", icon=":material/description:"),
        st.Page(P + "live_monitor.py", title="Live Monitor (preview)", icon=":material/sensors:"),
        st.Page(P + "settings.py", title="Settings", icon=":material/settings:"),
    ],
}
nav = st.navigation(pages, expanded=True)

with st.sidebar:
    st.markdown(f"**{APP_TITLE}**  \n<span class='small-muted'>v{APP_VERSION} · analytical layer above "
                f"the {SYSTEM_INFO['ems']} EMS</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("**Loaded batteries**")
    any_data = False
    for b in state.batteries():
        files = state.group_files(b.key)
        if files:
            any_data = True
        mark = "●" if files else "○"
        st.markdown(f"<span style='color:{b.color}'>{mark}</span> {b.name} · {b.nominal_kwh:,.0f} kWh "
                    f"<span class='small-muted'>({len(files)} file{'s' if len(files) != 1 else ''})</span>",
                    unsafe_allow_html=True)
    shared = state.group_files(state.SHARED)
    st.markdown(f"<span class='small-muted'>Shared plant files: {len(shared)}</span>", unsafe_allow_html=True)
    if st.session_state.get("synthetic_demo"):
        st.warning("Synthetic demo data loaded", icon="⚠️")
    st.divider()
    s = state.settings()
    st.markdown(f"<span class='small-muted'>Charging filter: P_charge > {s['charge_power_min_kw']:g} kW "
                f"and SOC < {s['soc_threshold']:g} %<br>Synchronization: {s['sync_method']}"
                f"{' @ ' + s['resample_interval'] if s['sync_method'] == 'resample' else ''}<br>"
                f"Split: {s['split_method']} {int(round((1 - s['test_size']) * 100))}/"
                f"{int(round(s['test_size'] * 100))}</span>", unsafe_allow_html=True)
    st.caption("Monitoring & analytics only – no battery control commands are issued.")

nav.run()
