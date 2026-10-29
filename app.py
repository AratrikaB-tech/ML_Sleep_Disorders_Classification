"""
Sleep Disorder Screening Assistant
One explainable model, three views: general user, clinician, workplace wellness.

This is a screening / decision-support demo built on a small SYNTHETIC dataset.
It is NOT a medical diagnosis tool.
"""
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
RANDOM_STATE = 42
MIN_GROUP = 5   # workplace view hides any group smaller than this

st.set_page_config(page_title="Sleep Screening Assistant", page_icon="🌙", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Loading the saved files
REQUIRED = ["sleep_model.joblib", "clean_data.csv", "model_metadata.json"]
missing = [f for f in REQUIRED if not (BASE / f).exists()]
if missing:
    st.error(
        "These files are missing from the app folder: " + ", ".join(missing) + ". "
        "Put app.py in the same folder as the files your notebook saved, then run it again."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(BASE / "sleep_model.joblib")


@st.cache_data
def load_clean():
    # keep_default_na=False so the label "None" is not read as a missing value
    return pd.read_csv(BASE / "clean_data.csv", keep_default_na=False)


@st.cache_data
def load_meta():
    with open(BASE / "model_metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_comparison():
    p = BASE / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return None


model = load_model()
clean_df = load_clean()
meta = load_meta()
comparison = load_comparison()

FEATURES = meta["features"]
CLASSES = meta["classes"]
NUM_COLS = meta["num_cols"]
CATS = meta["categories"]
RANGES = meta["numeric_ranges"]
DEPLOYED = meta["deployed_model"]
SUMMARY = meta.get("data_summary", {})

LABELS = {
    "gender": "Gender", "age": "Age", "occupation": "Occupation",
    "sleep_duration": "Sleep duration", "sleep_quality": "Sleep quality",
    "physical_activity": "Physical activity", "stress_level": "Stress level",
    "bmi_category": "Body weight category", "systolic": "Systolic blood pressure",
    "diastolic": "Diastolic blood pressure", "heart_rate": "Resting heart rate",
    "daily_steps": "Daily steps",
}
CLASS_NAMES = {"None": "No disorder pattern", "Insomnia": "Insomnia pattern",
               "Sleep Apnea": "Sleep apnea pattern"}
VITALS = ["systolic", "diastolic", "heart_rate"]


def cv_numbers():
    """Cross-validated accuracy / macro-F1 of the deployed model (if available)."""
    acc = meta.get("cv_results", {}).get(DEPLOYED)
    f1 = base_acc = None
    if comparison is not None and DEPLOYED in comparison.index:
        acc = comparison.loc[DEPLOYED, "CV Accuracy"] if "CV Accuracy" in comparison else acc
        f1 = comparison.loc[DEPLOYED, "CV F1 (macro)"] if "CV F1 (macro)" in comparison else None
        for name in comparison.index:
            if str(name).startswith("Baseline") and "CV Accuracy" in comparison:
                base_acc = comparison.loc[name, "CV Accuracy"]
    return acc, f1, base_acc


# The explanation function
def explain_person(pipe, row, background, n_values=40, seed=RANDOM_STATE):
    """Returns (predicted_class, probabilities, effects).
    effect > 0 : this input pushed the prediction TOWARD the predicted class
    effect < 0 : this input pushed it AWAY from the predicted class
    Method: swapping one input at a time with typical values from the data and
    seeing how the predicted probability changes."""
    classes = list(pipe.named_steps["clf"].classes_)
    base = pipe.predict_proba(row)[0]
    k = int(np.argmax(base))
    effects = {}
    for col in FEATURES:
        vals = background[col].sample(min(n_values, len(background)), random_state=seed).to_numpy()
        trial = pd.concat([row] * len(vals), ignore_index=True)
        trial[col] = vals
        effects[col] = base[k] - pipe.predict_proba(trial)[:, k].mean()
    eff = pd.Series(effects).sort_values(key=abs, ascending=False)
    return classes[k], dict(zip(classes, base)), eff


def predict_with_explanation(values):
    row = pd.DataFrame([values])[FEATURES]
    pred, probs, eff = explain_person(model, row, clean_df[FEATURES])
    return {"pred": pred, "probs": probs, "effects": eff, "values": values}


# Input widgets
def med(col, lo, hi, as_int=True):
    v = RANGES[col]["median"]
    v = min(max(v, lo), hi)
    return int(round(v)) if as_int else float(round(v, 1))


def core_inputs(prefix):
    """The lifestyle inputs everyone can answer."""
    c1, c2 = st.columns(2)
    v = {}
    with c1:
        v["age"] = st.number_input("Age", 18, 90, med("age", 18, 90), key=f"{prefix}_age")
        v["gender"] = st.selectbox("Gender", CATS["gender"], key=f"{prefix}_gender")
        v["occupation"] = st.selectbox("Occupation", CATS["occupation"], key=f"{prefix}_occ")
        v["bmi_category"] = st.selectbox(
            "Body weight category", CATS["bmi_category"], key=f"{prefix}_bmi",
            help="Normal, overweight or obese, as classed by BMI.")
        v["daily_steps"] = st.number_input(
            "Daily steps", 0, 30000, med("daily_steps", 0, 30000), step=500, key=f"{prefix}_steps")
    with c2:
        v["sleep_duration"] = st.slider(
            "Sleep per night (hours)", 3.0, 12.0, med("sleep_duration", 3, 12, False), 0.1,
            key=f"{prefix}_sd")
        v["sleep_quality"] = st.slider(
            "Sleep quality (1 = poor, 10 = excellent)", 1, 10, med("sleep_quality", 1, 10),
            key=f"{prefix}_sq")
        v["stress_level"] = st.slider(
            "Stress level (1 = low, 10 = high)", 1, 10, med("stress_level", 1, 10),
            key=f"{prefix}_stress")
        v["physical_activity"] = st.number_input(
            "Physical activity (minutes per day)", 0, 240, med("physical_activity", 0, 240),
            step=5, key=f"{prefix}_act")
    return v


def vital_inputs(prefix):
    c1, c2, c3 = st.columns(3)
    v = {}
    with c1:
        v["systolic"] = st.number_input("Systolic blood pressure", 80, 220, med("systolic", 80, 220),
                                        key=f"{prefix}_sys")
    with c2:
        v["diastolic"] = st.number_input("Diastolic blood pressure", 40, 140, med("diastolic", 40, 140),
                                         key=f"{prefix}_dia")
    with c3:
        v["heart_rate"] = st.number_input("Resting heart rate (beats/min)", 35, 160,
                                          med("heart_rate", 35, 160), key=f"{prefix}_hr")
    return v


def typical_vitals():
    return {c: int(round(RANGES[c]["median"])) for c in VITALS}


def out_of_range(values, skip=()):
    """Inputs outside the range the model saw during training."""
    flagged = []
    for c in NUM_COLS:
        if c in skip:
            continue
        lo, hi = RANGES[c]["min"], RANGES[c]["max"]
        if values[c] < lo or values[c] > hi:
            flagged.append(f"{LABELS[c]} ({values[c]}; training data covered {lo:g} to {hi:g})")
    return flagged


def confidence_text(p):
    acc, _, _ = cv_numbers()
    acc_txt = f"about {acc * 100:.0f}%" if acc is not None else "limited"
    if p >= 0.70:
        level = "fairly strong"
    elif p >= 0.50:
        level = "moderate"
    else:
        level = "weak"
    return level, acc_txt


# Shared display pieces
def probability_bars(probs):
    for cls, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        st.write(f"**{CLASS_NAMES.get(cls, cls)}**: {p * 100:.0f}%")
        st.progress(float(p))


def effects_chart(effects, values, skip=(), top=None):
    eff = effects.drop(labels=[s for s in skip if s in effects.index])
    if top:
        eff = eff.head(top)
    eff = eff.iloc[::-1]
    names = []
    for f in eff.index:
        val = values[f]
        names.append(f"{LABELS[f]} ({val})")
    colors = ["#2f7d6b" if x > 0 else "#c0603a" for x in eff.values]
    fig, ax = plt.subplots(figsize=(6.5, max(2.6, 0.38 * len(eff) + 0.8)))
    ax.barh(names, eff.values, color=colors)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Effect on the predicted result's probability")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Green bars pushed the model toward its result; orange bars pushed it away. "
               "These show what the model weighted, not what causes a sleep disorder.")


def footer():
    st.divider()
    st.caption(
        "Screening and decision-support demo only. It is not a diagnosis and does not replace "
        "a doctor. The model was trained on a small synthetic dataset "
        f"({SUMMARY.get('rows_after_dedup', 'few')} unique rows)."
    )


def sleep_tips(v):
    tips = []
    if v["sleep_duration"] < 7:
        tips.append("Most adults do best with about 7 or more hours of sleep. Try moving bedtime earlier by 15 minutes at a time.")
    if v["stress_level"] >= 7:
        tips.append("Your stress is high. A short wind-down routine before bed (no screens, light reading, slow breathing) has been reported to help many people.")
    if v["physical_activity"] < 30:
        tips.append("Regular movement during the day is linked with better sleep for many people. Even a daily walk counts.")
    if v["daily_steps"] < 5000:
        tips.append("Your daily steps are low. Adding a short walk after meals is an easy start.")
    tips.append("Keep a steady sleep and wake time, and avoid caffeine in the afternoon and evening.")
    return tips[:4]


# View 1: general user
def general_view():
    st.header("Quick sleep check")
    st.write("Answer a few questions and get a plain-language result with the main reasons behind it. "
             "This is a screening aid, not a diagnosis.")

    know = st.checkbox("I know my blood pressure and resting heart rate (optional, makes the result more reliable)",
                       key="g_know")
    with st.form("general_form"):
        values = core_inputs("g")
        if know:
            values.update(vital_inputs("g"))
        submitted = st.form_submit_button("Check my sleep pattern", type="primary")

    if submitted:
        assumed = [] if know else list(VITALS)
        if not know:
            values.update(typical_vitals())
        res = predict_with_explanation(values)
        res["assumed"] = assumed
        res["flags"] = out_of_range(values, skip=assumed)
        st.session_state["res_general"] = res

    res = st.session_state.get("res_general")
    if not res:
        footer()
        return

    st.subheader("Your result")
    pred, probs = res["pred"], res["probs"]
    top_p = probs[pred]
    level, acc_txt = confidence_text(top_p)

    if pred == "None":
        st.success(f"**{CLASS_NAMES[pred]}.** Your answers look closest to people without a sleep disorder.")
    else:
        st.warning(f"**{CLASS_NAMES[pred]}.** Your answers look closest to people with this pattern. "
                   "That is a reason to talk to a doctor, not a diagnosis.")
    st.write(f"Model confidence: **{level}** ({top_p * 100:.0f}%). "
             f"Across our testing the model was right {acc_txt} of the time, so any single result can be wrong.")

    if res["assumed"]:
        st.info("You did not enter blood pressure or heart rate, so typical values from the data were used. "
                "Those inputs matter to this model, so the result is less reliable without them.")
    for f in res["flags"]:
        st.warning(f"Outside the range the model learned from: {f}. Treat the result with extra caution.")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### How likely each pattern is")
        probability_bars(probs)
    with right:
        st.markdown("### Main reasons")
        eff = res["effects"].drop(labels=[a for a in res["assumed"] if a in res["effects"].index])
        pos = eff[eff > 0.005].head(3)
        neg = eff[eff < -0.005].head(1)
        target = CLASS_NAMES.get(pred, pred).lower()
        if len(pos) == 0:
            st.write("No single answer stood out. The result comes from many small effects together.")
        for f, e in pos.items():
            st.write(f"- **{LABELS[f]}** ({res['values'][f]}) pushed the model toward the {target}.")
        for f, e in neg.items():
            st.write(f"- **{LABELS[f]}** ({res['values'][f]}) pulled the other way.")
        st.caption("Reasons show what the model weighted in your answers. They are not medical causes.")

    st.markdown("### Things that may help")
    for t in sleep_tips(res["values"]):
        st.write(f"- {t}")
    st.write("If you often feel unrested, snore loudly, or stop breathing in your sleep, see a doctor "
             "whatever this tool says.")
    footer()


# View 2: clinician
def clinician_view():
    st.header("Clinician view")
    st.write("Please enter the patient's details. The tool returns class probabilities, the inputs that drove the result, "
             "and the model's known limits. Decision support only.")

    with st.form("clin_form"):
        st.markdown("### Lifestyle and demographics")
        values = core_inputs("c")
        st.markdown("### Vitals")
        values.update(vital_inputs("c"))
        submitted = st.form_submit_button("Run assessment", type="primary")

    if submitted:
        res = predict_with_explanation(values)
        res["flags"] = out_of_range(values)
        st.session_state["res_clin"] = res

    res = st.session_state.get("res_clin")
    if not res:
        footer()
        return

    pred, probs, eff, vals = res["pred"], res["probs"], res["effects"], res["values"]
    st.subheader("Assessment")
    m1, m2, m3 = st.columns(3)
    m1.metric("Most likely pattern", CLASS_NAMES.get(pred, pred))
    m2.metric("Probability", f"{probs[pred] * 100:.0f}%")
    ordered = sorted(probs.items(), key=lambda kv: -kv[1])
    margin = ordered[0][1] - ordered[1][1]
    m3.metric("Margin over next class", f"{margin * 100:.0f} pts")

    for f in res["flags"]:
        st.warning(f"Outside training range: {f}.")
    if margin < 0.15:
        st.warning("The top two classes are close. Please treat this as inconclusive and rely on clinical judgement.")

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown("### Class probabilities")
        prob_df = pd.DataFrame({"Pattern": [CLASS_NAMES.get(c, c) for c, _ in ordered],
                                "Probability": [f"{p * 100:.1f}%" for _, p in ordered]})
        st.dataframe(prob_df, hide_index=True, width="stretch")
    with right:
        st.markdown("### Inputs that drove this result")
        effects_chart(eff, vals)

    with st.expander("Patient values compared with the training data"):
        rows = []
        for c in NUM_COLS:
            rows.append({"Input": LABELS[c], "Patient": vals[c],
                         "Training median": RANGES[c]["median"],
                         "Training range": f"{RANGES[c]['min']:g} to {RANGES[c]['max']:g}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("### Model reliability")
    acc, f1, base_acc = cv_numbers()
    r1, r2, r3 = st.columns(3)
    r1.metric("Model", DEPLOYED)
    r2.metric("Cross-validated accuracy", f"{acc * 100:.0f}%" if acc is not None else "n/a")
    r3.metric("Cross-validated macro F1", f"{f1:.2f}" if f1 is not None else "n/a")
    note = (f"- Trained on {SUMMARY.get('rows_after_dedup', '?')} unique records "
            f"({SUMMARY.get('rows_original', '?')} rows before removing duplicates). The data is synthetic.\n"
            "- Scores come from repeated cross-validation on de-duplicated data. Reference studies that kept "
            "duplicates can report much higher accuracy because test rows were copies of training rows.\n")
    if base_acc is not None:
        note += f"- A model that always guesses the most common class scores about {base_acc * 100:.0f}% accuracy.\n"
    if SUMMARY.get("naive_rf_accuracy") is not None:
        note += (f"- In our own check, a random forest scored {SUMMARY['naive_rf_accuracy'] * 100:.0f}% when duplicates "
                 "leaked into the test set, which is not a fair figure.\n")
    note += ("- Explanations come from swapping one input at a time with typical values and watching the probability "
             "change. They describe the model, not the patient's physiology.")
    st.markdown(note)
    footer()


# View 3: workplace wellness (aggregate only)
KAGGLE_RENAME = {
    "Gender": "gender", "Age": "age", "Occupation": "occupation",
    "Sleep Duration": "sleep_duration", "Quality of Sleep": "sleep_quality",
    "Physical Activity Level": "physical_activity", "Stress Level": "stress_level",
    "BMI Category": "bmi_category", "Heart Rate": "heart_rate", "Daily Steps": "daily_steps",
}


def standardise_upload(df):
    df = df.copy()
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        df["systolic"] = pd.to_numeric(bp[0], errors="coerce")
        df["diastolic"] = pd.to_numeric(bp[1], errors="coerce")
    df = df.rename(columns=KAGGLE_RENAME)
    if "bmi_category" in df.columns:
        df["bmi_category"] = df["bmi_category"].astype(str).str.strip().replace({"Normal Weight": "Normal"})
    for c in ("gender", "occupation"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def template_csv():
    row = {c: clean_df[c].iloc[0] for c in FEATURES}
    return pd.DataFrame([row])[FEATURES].to_csv(index=False)


def group_table(df, by, label):
    g = df.groupby(by, observed=True)["flag"].agg(["size", "mean"])
    shown = g[g["size"] >= MIN_GROUP]
    hidden = int((g["size"] < MIN_GROUP).sum())
    st.markdown(f"### {label}")
    if shown.empty:
        st.write(f"All groups have fewer than {MIN_GROUP} people, so none can be shown.")
        return
    chart = (shown["mean"] * 100).rename("People with a predicted sleep-disorder pattern (%)")
    st.bar_chart(chart)
    st.caption(f"Groups shown have at least {MIN_GROUP} people. " +
               (f"{hidden} smaller group(s) hidden to protect privacy." if hidden else ""))


def hr_view():
    st.header("Workplace wellness view")
    st.write("See sleep-risk patterns across a team, never for individuals. Results are grouped, "
             f"and any group smaller than {MIN_GROUP} people is hidden.")

    source = st.radio("Data source", ["Demo data (the cleaned dataset)", "Upload a CSV"], horizontal=True)
    data = None
    if source.startswith("Demo"):
        data = clean_df[FEATURES].copy()
        st.info("Demo mode. These rows were also used to train the model, so the predictions are flattering "
                "and are shown only to demonstrate the view.")
    else:
        st.download_button("Download CSV template", template_csv(), "workforce_template.csv", "text/csv")
        up = st.file_uploader("Upload a CSV with one row per employee (anonymised)", type="csv")
        if up is not None:
            try:
                raw = pd.read_csv(up, keep_default_na=False, na_values=[""])
            except Exception as e:
                st.error(f"Could not read that file: {e}")
                raw = None
            if raw is not None:
                df = standardise_upload(raw)
                absent = [c for c in FEATURES if c not in df.columns]
                if absent:
                    st.error("Missing columns: " + ", ".join(absent) +
                             ". Use the template, or the original Kaggle column names.")
                else:
                    df = df[FEATURES].copy()
                    for c in NUM_COLS:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    before = len(df)
                    df = df.dropna().reset_index(drop=True)
                    if len(df) < before:
                        st.warning(f"{before - len(df)} row(s) with missing or invalid values were skipped.")
                    data = df

    if data is None:
        footer()
        return
    if len(data) < MIN_GROUP:
        st.error(f"At least {MIN_GROUP} people are needed to show any results.")
        return

    preds = model.predict(data[FEATURES])
    out = data.copy()
    out["pred"] = preds
    out["flag"] = (out["pred"] != "None").astype(float)

    st.subheader("Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("People analysed", len(out))
    k2.metric("Predicted any sleep-disorder pattern", f"{out['flag'].mean() * 100:.0f}%")
    k3.metric("Average sleep per night", f"{out['sleep_duration'].mean():.1f} h")

    dist = out["pred"].map(lambda c: CLASS_NAMES.get(c, c)).value_counts()
    st.markdown("##### Predicted pattern mix")
    st.bar_chart(dist)

    out["Stress band"] = pd.cut(out["stress_level"], [0, 3, 6, 10],
                                labels=["Low (1-3)", "Moderate (4-6)", "High (7-10)"])
    out["Sleep band"] = pd.cut(out["sleep_duration"], [0, 6, 7, 8, 24], right=False,
                               labels=["Under 6 h", "6 to 7 h", "7 to 8 h", "8 h or more"])
    c1, c2 = st.columns(2)
    with c1:
        group_table(out, "Stress band", "By stress level")
    with c2:
        group_table(out, "Sleep band", "By sleep duration")
    group_table(out, "occupation", "By occupation")

    st.markdown("### What teams can do with this")
    st.write("- If high-stress groups show more sleep-disorder patterns, look at workload, shift length and after-hours messages.\n"
             "- If short sleepers cluster in particular roles, review start times and shift patterns.\n"
             "- Offer voluntary, confidential access to sleep screening and a doctor, and never use results to judge individuals.")
    st.caption("No individual rows are displayed, and predictions are not stored. Group results are statistical "
               "patterns from a screening model with limited accuracy.")
    footer()


# Landing page and routing
def set_persona(p):
    st.session_state["persona"] = p


def landing():
    st.title("Sleep Screening Assistant")
    st.write("A screening tool that tells you what it predicts and why. Pick the view that fits you.")
    c1, c2, c3 = st.columns(3)
    cards = [
        (c1, "general", "I want to check my own sleep",
         "A few simple questions, a plain-language result, and the main reasons behind it."),
        (c2, "clinician", "I am a clinician",
         "Class probabilities, input effects, and the model's reliability and limits."),
        (c3, "hr", "I look after a team's wellbeing",
         "Anonymous, grouped patterns across a workforce. Individuals are never shown."),
    ]
    for col, key, title, desc in cards:
        with col:
            with st.container(border=True):
                st.subheader(title)
                st.write(desc)
                st.button("Open this view", key=f"pick_{key}", on_click=set_persona, args=(key,),
                          width="stretch")
    footer()


if "persona" not in st.session_state:
    st.session_state["persona"] = None

with st.sidebar:
    st.markdown("### Sleep Screening Assistant")
    if st.session_state["persona"] is not None:
        st.button("Change view", key="change_view", on_click=set_persona, args=(None,))
    with st.expander("About this tool"):
        acc, f1, _ = cv_numbers()
        st.write(f"Model: {DEPLOYED}")
        if acc is not None:
            st.write(f"Cross-validated accuracy: about {acc * 100:.0f}%")
        st.write(f"Trained on {SUMMARY.get('rows_after_dedup', '?')} unique records of a synthetic dataset "
                 "(duplicates removed before testing).")
        st.write("Screening and decision support only. Not a diagnosis.")

persona = st.session_state["persona"]
if persona == "general":
    general_view()
elif persona == "clinician":
    clinician_view()
elif persona == "hr":
    hr_view()
else:
    landing()