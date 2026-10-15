"""End-to-end UI checks using Streamlit's AppTest harness."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TIMEOUT = 180


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def loaded():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    button(at, "Use bundled sample dataset").click().run()
    assert not at.exception
    return at


def test_sample_dataset_loads(loaded):
    assert any("6,745 requests" in s.value for s in loaded.success)


def test_solve_both_models(loaded):
    at = loaded
    button(at, "Solve").click().run()
    assert not at.exception and not at.error
    assert any(m.label == "Unmet requests" for m in at.metric)
    at.radio(key="model_choice").set_value("Model 2 – Goal Programming").run()
    button(at, "Solve").click().run()
    assert not at.exception and not at.error
    assert any(m.label == "G2 target shortfall share" for m in at.metric)


def test_sensitivity_and_comparison(loaded):
    at = loaded
    for label in ["Run capacity sensitivity", "Run productivity sensitivity",
                  "Run target sensitivity", "Compute shadow prices", "Run comparison"]:
        button(at, label).click().run()
        assert not at.exception, label
        assert not at.error, label
    assert any(m.label == "Periods where capacity binds" for m in at.metric)


def test_bad_sweep_input_shows_error(loaded):
    at = loaded
    at.text_input[0].set_value("abc").run()
    button(at, "Run capacity sensitivity").click().run()
    assert not at.exception
    assert any("Invalid list" in e.value for e in at.error)


def test_stale_results_hidden_after_data_change(loaded):
    at = loaded
    button(at, "Solve").click().run()
    button(at, "Run comparison").click().run()
    assert any(m.label == "Unmet requests" and m.value == "2,239" for m in at.metric)
    at.radio(key="input_mode").set_value("Enter values manually").run()
    assert not at.exception
    assert not any(m.value == "2,239" for m in at.metric)
    assert not any("2,239" in str(df.value.to_dict()) for df in at.dataframe)
    assert any("Inputs changed" in i.value for i in at.info)


def test_params_persist_across_model_switch(loaded):
    at = loaded
    at.number_input(key="wgt_w_A").set_value(5.0).run()
    at.radio(key="model_choice").set_value("Model 2 – Goal Programming").run()
    at.slider(key="wgt_g_C").set_value(95).run()
    assert at.session_state["w_A"] == 5.0
    at.radio(key="model_choice").set_value("Model 1 – Integer Programming").run()
    assert at.number_input(key="wgt_w_A").value == 5.0
    assert at.session_state["g_C"] == 95


def test_kpis_include_mean_period_gap(loaded):
    at = loaded
    button(at, "Solve").click().run()
    assert any(m.label == "Mean gap per period" for m in at.metric)


def test_manual_edits_restored_after_mode_switch():
    import pandas as pd
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio(key="input_mode").set_value("Enter values manually").run()
    edited = pd.DataFrame([("A", 3, 4, 2), ("B", 5, 1, 3)],
                          columns=["Period", "City demand", "Airport demand", "Active drivers"])
    at.radio(key="input_mode").set_value("Upload CSV").run()
    at.session_state["manual_saved"] = edited  # what the editor held before the switch
    at.radio(key="input_mode").set_value("Enter values manually").run()
    assert not at.exception
    assert any("2 valid period" in s.value for s in at.success)


def test_manual_mode_solves():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio(key="input_mode").set_value("Enter values manually").run()
    assert not at.exception
    assert any("4 valid period" in s.value for s in at.success)
    button(at, "Solve").click().run()
    assert not at.exception and not at.error
    button(at, "Run comparison").click().run()
    assert not at.exception and not at.error
