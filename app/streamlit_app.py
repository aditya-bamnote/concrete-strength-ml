"""Streamlit web app: enter a concrete mix, get the predicted compressive strength.

Run from the project folder:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so `import src...` works however the app is launched

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import FEATURES, LABELS
from src.predict import (DEFAULT_PRESET, InputValidationError, PRESETS, load_artifacts,
                         predict_strength, strength_curve)

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e5e4e0"

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    model, meta = get_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

R = meta["feature_ranges"]
STEP = {"cement": 5.0, "slag": 5.0, "fly_ash": 5.0, "water": 1.0,
        "superplasticizer": 0.5, "coarse_agg": 5.0, "fine_agg": 5.0}
GROUPS = [
    ("Binder materials", ["cement", "slag", "fly_ash"]),
    ("Water and admixture", ["water", "superplasticizer"]),
    ("Aggregates", ["coarse_agg", "fine_agg"]),
    ("Curing", ["age"]),
]


# ---------------------------------------------------------------- preset handling
def apply_preset():
    preset = PRESETS.get(st.session_state["preset"])
    if preset:
        for name, value in preset.items():
            st.session_state[f"in_{name}"] = value
        st.session_state.pop("result", None)


if "in_cement" not in st.session_state:
    for name, value in PRESETS[DEFAULT_PRESET].items():
        st.session_state[f"in_{name}"] = value


# ---------------------------------------------------------------- header
st.title("🏗️ Concrete Compressive Strength Predictor")
st.caption(
    "Computational Intelligence project · Predicts the compressive strength (MPa) of concrete "
    "from its mix proportions and curing age · Supports SDG 9 (Industry, Innovation & Infrastructure)"
)

tab_predict, tab_model = st.tabs(["Predict", "About the model"])

# ---------------------------------------------------------------- predict tab
with tab_predict:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("1. Describe the mix")
        st.selectbox("Start from an example (optional)", list(PRESETS), key="preset", on_change=apply_preset,
                     help="Loads typical values. You can edit any of them afterwards.")
        with st.form("mix_form"):
            for title, names in GROUPS:
                st.markdown(f"**{title}**")
                cols = st.columns(len(names))
                for col, name in zip(cols, names):
                    lo, hi = R[name]["min"], R[name]["max"]
                    with col:
                        if name == "age":
                            st.number_input(LABELS[name], min_value=int(lo), max_value=int(hi), step=1,
                                            key=f"in_{name}", help=f"Allowed range: {int(lo)} to {int(hi)} days")
                        else:
                            st.number_input(LABELS[name].replace("Blast furnace slag", "Slag"),
                                            min_value=float(lo), max_value=float(hi),
                                            step=STEP[name], key=f"in_{name}",
                                            help=f"Allowed range: {lo:g} to {hi:g} kg/m³ (range of the training data)")
            submitted = st.form_submit_button("Predict strength", type="primary")

    if submitted:
        values = {name: st.session_state[f"in_{name}"] for name in FEATURES}
        try:
            st.session_state["result"] = predict_strength(model, meta, values)
            st.session_state["curve"] = strength_curve(model, meta, values)
            st.session_state.pop("error", None)
        except InputValidationError as exc:
            st.session_state["error"] = exc.errors
            st.session_state.pop("result", None)
        except Exception as exc:  # noqa: BLE001 - never show a raw traceback to the user
            st.session_state["error"] = [f"Prediction failed unexpectedly: {exc}"]
            st.session_state.pop("result", None)

    with right:
        st.subheader("2. Predicted strength")
        if "error" in st.session_state:
            for message in st.session_state["error"]:
                st.error(message)
        elif "result" not in st.session_state:
            st.info("Enter the mix on the left (or pick an example) and press **Predict strength**.")
        else:
            res, curve = st.session_state["result"], st.session_state["curve"]
            for message in res["warnings"]:
                st.warning(message)
            c1, c2 = st.columns(2)
            c1.metric("Predicted compressive strength", f"{res['strength_mpa']:.1f} MPa")
            c2.metric("Typical range (90 % of cases)", f"{res['low_mpa']:.1f} – {res['high_mpa']:.1f} MPa")
            st.markdown(f"**Indicative category:** {res['category']}  \n"
                        "<small>A rough label only – not a design-code concrete grade.</small>",
                        unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.metric("Water / cement ratio", f"{res['water_cement']:.2f}")
            d2.metric("Water / binder ratio", f"{res['water_binder']:.2f}",
                      help="Binder = cement + slag + fly ash. Lower ratios generally give stronger concrete.")

            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(curve["age"], curve["strength_mpa"], color=BLUE, lw=2, marker="o", ms=5)
            this_age = res["inputs"]["age"]
            ax.scatter([this_age], [res["strength_mpa"]], s=90, color=ORANGE, zorder=5, label="your mix")
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 7, 28, 90, 365])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("curing age (days, log scale)", color=MUTED)
            ax.set_ylabel("predicted strength (MPa)", color=MUTED)
            ax.set_title("Same mix at other curing ages", loc="left", fontsize=10, fontweight="600")
            ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("The curve is the model's own estimate learned from the data, not a physical formula. "
                       "Ages beyond about 90 days rely on fewer, different mixes in the data, so treat them with extra care.")

    st.divider()
    st.caption("⚠️ Screening tool only. Predictions are estimates from laboratory data (UCI, Yeh 1998) and are "
               "reliable only for mixes similar to the training data. Always confirm real designs by testing "
               "and the relevant standards.")

# ---------------------------------------------------------------- model tab
with tab_model:
    m = meta["test_metrics"]
    st.subheader(f"Model: {meta['model_name']} (selected by cross-validation)")
    a, b, c, d = st.columns(4)
    a.metric("Test R²", f"{m['R2']:.3f}")
    b.metric("Test MAE", f"{m['MAE']:.2f} MPa")
    c.metric("Test RMSE", f"{m['RMSE']:.2f} MPa")
    d.metric("Test mixes", meta["n_test"])
    st.write(
        f"Trained on {meta['n_train']} mixes and evaluated once on {meta['n_test']} unseen mixes. "
        f"The deployed model was then refitted on all {meta['n_rows_total']} unique mixes. "
        f"The ± range shown with each prediction comes from cross-validation residuals; on the test set it contained "
        f"{meta['residual_band_90']['test_coverage']:.0%} of the actual values."
    )
    with st.expander("Tuned hyper-parameters"):
        st.json(meta["best_params"])
    with st.expander("Allowed input ranges (from the training data)"):
        st.dataframe(pd.DataFrame(R).T.rename_axis("input").round(2))

    figs = ROOT / "reports" / "figures"
    for fname, caption in [("07_model_comparison.png", "Model comparison (cross-validated R²)"),
                           ("08_predicted_vs_actual.png", "Predicted vs. actual strength on the test set"),
                           ("10_feature_importance.png", "Which inputs matter most")]:
        if (figs / fname).exists():
            st.image(str(figs / fname), caption=caption)
    st.caption(f"Data: {meta['dataset']}")
