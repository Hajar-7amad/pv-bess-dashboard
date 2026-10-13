"""Headless smoke test of every Streamlit page with the SYNTHETIC demo data.

Run: python -m pytest -q tests/test_app.py
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["overview", "data_explorer", "plot_builder", "data_quality", "battery_analysis", "charging_analysis",
         "battery_comparison", "machine_learning", "shap_dashboard", "reports", "live_monitor", "settings"]


def _app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    at.switch_page("pages/data_import.py").run()
    btn = [b for b in at.button if b.label.startswith("Load SYNTHETIC")][0]
    btn.click().run()
    assert at.session_state["synthetic_demo"]
    return at


def test_all_pages_render():
    at = _app()
    for p in PAGES:
        at.switch_page(f"pages/{p}.py").run()
        assert not at.exception, f"{p}: {at.exception}"


def test_train_and_shap():
    at = _app()
    at.switch_page("pages/machine_learning.py").run()
    [b for b in at.button if b.label.startswith("Train all")][0].click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["models"]) == 2
    for p in ["machine_learning", "shap_dashboard", "battery_comparison", "reports"]:
        at.switch_page(f"pages/{p}.py").run()
        assert not at.exception, f"{p}: {at.exception}"
    at.switch_page("pages/reports.py").run()
    [b for b in at.button if b.label == "Generate report"][0].click().run()
    assert not at.exception, at.exception
    assert "<html" in at.session_state["report_html"]
