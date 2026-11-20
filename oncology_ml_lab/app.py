"""Friendly, public educational explorer for a fixed breast-cancer benchmark.

This app does not collect patient data and is not a medical diagnosis tool.
"""
from pathlib import Path
import json

import joblib
import streamlit as st

from src.common import load_data
from src.demo import class_name, predicted_label, sample_groups

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Oncology ML Lab | Understand the AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  :root { color-scheme: light; }
  .stApp { background: #f5f8fc; color: #12243a; }
  .block-container { max-width: 1120px; padding-top: 1.3rem; padding-bottom: 3rem; }
  h1, h2, h3 { letter-spacing: -.025em; color: #132b47; }
  .hero {
    background: linear-gradient(125deg,#112c4c 0%,#184d77 68%,#087d87 100%);
    color: white; padding: 32px 35px; border-radius: 22px;
    box-shadow: 0 14px 38px rgba(13,51,87,.13); margin-bottom: 16px;
  }
  .hero .eyebrow { font-size: .76rem; text-transform: uppercase; font-weight: 800;
    letter-spacing: .14em; color: #a4f5ef; margin-bottom: 8px; }
  .hero h1 { font-size: clamp(1.85rem, 4vw, 2.85rem); color: #fff;
    margin: 0 0 9px 0; line-height: 1.15; }
  .hero p { font-size: 1.06rem; color: #eef9ff; max-width: 760px;
    margin: 0; line-height: 1.6; }
  .intro { background: #fff; border: 1px solid #dce8f2; border-radius: 14px;
    padding: 16px 19px; margin: 10px 0 18px; font-size: 1rem; line-height: 1.6; }
  .safety { padding: 12px 16px; border-radius: 12px; border: 1px solid #f2d59c;
    background: #fff7e8; font-size: .9rem; color: #6b4610; margin: 12px 0 18px; }
  .guide { background: white; border: 1px solid #dbe7f1; border-radius: 17px;
    padding: 16px 20px; min-height: 128px; }
  .step-index { display: inline-block; background: #e3f3ff; color: #12558a;
    font-size: .8rem; font-weight: 900; padding: 3px 9px; border-radius: 8px; margin-bottom: 8px; }
  .guide h3 { font-size: 1rem; margin: 0 0 5px; }
  .guide p { color: #415a72; font-size: .91rem; line-height: 1.45; margin: 0; }
  .results-card { background: white; border: 2px solid #dce8f4; border-radius: 19px;
    padding: 20px 22px; min-height: 142px; }
  .results-card.correct-label { border-color: #0d8190; }
  .results-card.malignant-label { border-color: #d47a5e; }
  .result-eyebrow { color: #51677f; font-size: .85rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: .055em; }
  .result-value { font-size: clamp(1.45rem,3.3vw,2rem); font-weight: 850;
    margin: 10px 0 5px; color: #13304c; }
  .result-help { color: #526880; font-size: .88rem; }
  .outcome { margin: 16px 0 8px; border-radius: 14px; padding: 16px 20px;
    font-size: 1.06rem; font-weight: 800; }
  .outcome.success { background: #dcf7ed; border: 1px solid #70ccb2; color: #0d6146; }
  .outcome.error { background: #fff0eb; border: 1px solid #f0a990; color: #8e3524; }
  .small-note { font-size: .88rem; line-height: 1.6; color: #526980; margin: 7px 0 13px; }
  .score-card { background: white; border: 1px solid #dce8f4; border-radius: 16px;
    padding: 19px 22px; margin: 15px 0; }
  .score-card strong { color: #13304c; }
  .score-bar { position: relative; background: #e2edf5; height: 20px;
    border-radius: 20px; overflow: visible; margin-top: 18px; }
  .score-fill { height: 100%; background: linear-gradient(90deg,#28b1b0,#ec946c);
    border-radius: 20px; }
  .threshold-line { position: absolute; top: -7px; bottom: -7px;
    border-left: 3px dashed #152f53; }
  .score-ticks { display: flex; justify-content: space-between; margin-top: 13px;
    font-size: .79rem; color: #456178; }
  .little-stat { background: white; border: 1px solid #dce8f4; border-radius: 16px;
    padding: 18px 19px; min-height: 112px; }
  .little-stat .n { font-size: 1.8rem; color: #153c60; font-weight: 850; }
  .little-stat .caption { font-size: .9rem; color: #526980; line-height: 1.4; }
  .explain-line { border-left: 3px solid #008f97; background: #edfafd;
    padding: 12px 16px; border-radius: 0 9px 9px 0; margin: 10px 0; }
  div.stButton > button { border-radius: 11px; border: 1px solid #aac9de;
    font-weight: 700; min-height: 2.8rem; }
  div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 750; }
  [data-testid="stSlider"] { background: white; padding: 15px 18px 6px;
    border: 1px solid #dce8f4; border-radius: 14px; }
  @media(max-width: 650px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: .7rem; }
    .hero { padding: 23px 21px; border-radius: 16px; }
    .guide, .results-card, .score-card { padding: 16px; }
    .score-ticks { flex-wrap: wrap; gap: 9px; }
  }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = ROOT / "models" / "model.joblib"
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


@st.cache_data(show_spinner=False)
def load_benchmark():
    return load_data()


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics_path = ROOT / "reports" / "metrics.json"
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


st.markdown("""
<div class="hero">
  <div class="eyebrow">🔬 Oncology ML Lab · Beginner-friendly AI demo</div>
  <h1>See how AI classifies breast cancer research samples</h1>
  <p>Choose an existing research sample. The computer examines 30 measurements
  and guesses <b>Benign (non-cancerous)</b> or <b>Malignant (cancerous)</b>.
  You can compare its guess with the answer already known in the dataset.</p>
</div>
<div class="intro">
  <b>Think of it like a school exam:</b> the AI studies examples first,
  then we check its answers on samples it did not study.
  <b>You can see both the real answer and the AI's guess below.</b>
</div>
<div class="safety"><b>⚠️ For learning only.</b> This app explores historical research data;
 it is <b>not</b> a medical test or diagnosis. It cannot tell anyone whether they have cancer.</div>
""", unsafe_allow_html=True)

artifact = load_model()
if artifact is None:
    st.error("The saved model was not found. From the project folder, run `python -m src.train` first.")
    st.stop()

X, y = load_benchmark()
cols = artifact["columns"]
threshold = float(artifact["threshold"])
model = artifact["pipeline"]

if not all(c in X.columns for c in cols):
    st.error("The dataset does not match the model's expected measurements.")
    st.stop()

splits = sample_groups(X, y)
metrics = load_metrics()

# Precompute scores so the user can jump directly to useful *held-out* examples.
scores = model.predict_proba(X[cols])[:, 1]
test_ids = splits["test"]
correct_benign = next((j for j, rid in enumerate(test_ids)
    if y.loc[rid] == 0 and predicted_label(scores[rid], threshold) == 0), None)
correct_malignant = next((j for j, rid in enumerate(test_ids)
    if y.loc[rid] == 1 and predicted_label(scores[rid], threshold) == 1), None)
missed_malignant = next((j for j, rid in enumerate(test_ids)
    if y.loc[rid] == 1 and predicted_label(scores[rid], threshold) == 0), None)

st.subheader("👣 Start here: 3 easy steps")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="guide"><span class="step-index">STEP 1</span><h3>Pick one sample</h3><p>Each sample is one row of saved research measurements. No names or patient details.</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="guide"><span class="step-index">STEP 2</span><h3>See two answers</h3><p>Compare the <b>known answer</b> from the dataset with the <b>AI prediction</b>.</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="guide"><span class="step-index">STEP 3</span><h3>See if AI was right</h3><p>Green means the answers match. Orange means the AI made a mistake.</p></div>', unsafe_allow_html=True)

explore_tab, learn_tab, results_tab = st.tabs([
    "🔎 Explore a sample", "🧠 How the AI learned", "📊 Results and charts"
])

with explore_tab:
    st.markdown("### ① Pick a sample")
    st.write("Every sample has 30 numerical cell measurements and a known label. Let's see what AI says about it!")

    show_all = st.toggle(
        "Explore all 569 records instead of just the 114 final test records",
        value=False,
        help="The final test group contains records the model did not study. The full dataset also contains training and validation records.",
    )
    available = list(map(int, X.index)) if show_all else test_ids
    selection_key = "all_case_position" if show_all else "test_case_position"
    if selection_key not in st.session_state:
        st.session_state[selection_key] = 1

    if not show_all:
        st.caption("Showing **114 test samples** that were held out from training and validation. Great for seeing real model mistakes.")
        e1, e2, e3 = st.columns(3)
        with e1:
            if correct_benign is not None and st.button("✅ Correct benign", use_container_width=True):
                st.session_state[selection_key] = correct_benign + 1
        with e2:
            if correct_malignant is not None and st.button("✅ Correct malignant", use_container_width=True):
                st.session_state[selection_key] = correct_malignant + 1
        with e3:
            if missed_malignant is not None and st.button("❌ See an AI mistake", use_container_width=True):
                st.session_state[selection_key] = missed_malignant + 1

    n = st.slider(
        f"Choose sample number (1 to {len(available)})",
        min_value=1, max_value=len(available), key=selection_key,
    )
    rid = available[n-1]
    true_label = int(y.loc[rid])
    score = float(scores[rid])
    guess_label = predicted_label(score, threshold)
    true_name = class_name(true_label)
    guess_name = class_name(guess_label)
    true_explanation = "Cancerous" if true_label == 1 else "Non-cancerous"
    guess_explanation = "Cancerous" if guess_label == 1 else "Non-cancerous"
    match = guess_label == true_label

    st.markdown(
        f'<div class="intro"><b>You are looking at sample {n} of {len(available)}.</b> '
        f'<br><span class="small-note">Original dataset row: {rid + 1} of 569. '
        f'This is a sample number, <b>not a person’s name or patient ID</b>.</span></div>',
        unsafe_allow_html=True,
    )
    a, b = st.columns(2)
    with a:
        st.markdown(
            f'<div class="results-card correct-label"><div class="result-eyebrow">📋 Actual answer</div>'
            f'<div class="result-value">{true_name}</div>'
            f'<div class="result-help">{true_explanation}. The answer recorded in the research dataset.</div></div>',
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f'<div class="results-card {"correct-label" if match else "malignant-label"}">'
            f'<div class="result-eyebrow">🤖 AI prediction</div>'
            f'<div class="result-value">{guess_name}</div>'
            f'<div class="result-help">{guess_explanation}. What AI predicts from the 30 measurements.</div></div>',
            unsafe_allow_html=True,
        )
    if match:
        st.markdown('<div class="outcome success">✅ Correct! The AI prediction matches the known answer.</div>', unsafe_allow_html=True)
    elif true_label == 1:
        st.markdown('<div class="outcome error">❌ The AI made a mistake: this malignant sample was predicted benign.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="outcome error">❌ The AI made a mistake: this benign sample was predicted malignant.</div>', unsafe_allow_html=True)

    st.markdown("### ② How did it choose that answer?")
    st.markdown(
        f'<div class="score-card">'
        f'<div><strong>Model score for “Malignant”: {score * 100:.1f} out of 100</strong></div>'
        f'<div class="small-note">Rule learned using the validation samples: '
        f'if the score is <b>{threshold * 100:.1f} or higher</b>, predict Malignant. '
        f'Otherwise, predict Benign.</div>'
        f'<div class="score-bar"><div class="score-fill" style="width:{score * 100:.3f}%"></div>'
        f'<div class="threshold-line" style="left:{threshold * 100:.3f}%"></div></div>'
        f'<div class="score-ticks"><span>0 — lower malignant score</span><span>Dashed line = {threshold * 100:.1f}</span><span>100 — higher</span></div>'
        f'<div class="small-note">⚠️ This is a <b>model score</b>, not a clinically validated cancer risk or certainty.</div>'
        f'</div>', unsafe_allow_html=True,
    )

    st.markdown("### ③ What information did AI read?")
    st.write("The computer uses **30 measurements of cell nuclei**, such as size, texture and shape. It does not look at patient names or photographs.")
    showcase = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area"]
    subset = [c for c in showcase if c in X.columns]
    if subset:
        measurement_cols = st.columns(len(subset))
        for box, feature in zip(measurement_cols, subset):
            box.metric(feature.replace("_", " ").title(), f"{X.loc[rid, feature]:.2f}")
    with st.expander("🔍 See all 30 measurements (optional)"):
        sample_table = X.loc[rid, cols]
        st.dataframe(
            sample_table.rename_axis("Measurement").reset_index(name="Value").assign(
                Measurement=lambda d: d["Measurement"].str.replace("_", " ").str.title()
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("The numbers are research measurements. They are not a diagnosis by themselves.")

    if show_all:
        group = next(name for name, ids in splits.items() if rid in ids)
        group_note = {
            "train": "This row was used to train the model, so its prediction is not an independent test.",
            "validation": "This row helped choose the decision threshold, so it is not an independent final test.",
            "test": "This row was held out as part of the final test group.",
        }[group]
        st.info(f"**Dataset group: {group.title()}.** {group_note}")
    else:
        st.caption("You're viewing a final-test sample. These 114 records were not used to train the model or select its threshold.")

with learn_tab:
    st.markdown("### How did we teach the AI?")
    st.markdown('<div class="explain-line"><b>Big idea:</b> Show the computer many examples with correct answers. It learns patterns, then we test it on samples it has not studied.</div>', unsafe_allow_html=True)
    st.markdown("#### Step 1 — Start with 569 research samples")
    st.write("Each sample is one row with 30 measurements and a known answer: **Benign** or **Malignant**.")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown('<div class="little-stat"><div class="n">341</div><div class="caption"><b>Training</b><br>Examples used to teach the model.</div></div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="little-stat"><div class="n">114</div><div class="caption"><b>Validation</b><br>Examples used to choose the decision rule.</div></div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="little-stat"><div class="n">114</div><div class="caption"><b>Final test</b><br>New examples used to measure mistakes.</div></div>', unsafe_allow_html=True)
    st.markdown("#### Step 2 — Try two ways to learn")
    st.write("We compared **Logistic Regression** and **Random Forest**. Both learned from the training samples. We checked them repeatedly using 5-fold cross-validation.")
    st.markdown("#### Step 3 — Choose the model and the decision rule")
    st.write(f"Logistic Regression performed better in cross-validation, so the code selected it. Using validation samples, it chose the **{threshold * 100:.1f}** score cutoff.")
    st.markdown("#### Step 4 — Take the final test")
    st.write("The AI then predicted labels for the 114 samples held aside for testing. We compared those predictions with their known answers.")
    st.info("The app **uses a saved, already-trained model**. Moving the slider does **not** train the model again.")

with results_tab:
    st.markdown("### How well did the AI do on the final test?")
    if metrics:
        matrix = metrics["confusion_matrix_rows_true_0_1"]
        correct_count = int(matrix[0][0] + matrix[1][1])
        test_count = int(metrics["test_count"])
        errors = test_count - correct_count
        misses = int(matrix[1][0])
        cols_stats = st.columns(3)
        for column, big, small in zip(
            cols_stats,
            [f"{correct_count} / {test_count}", str(errors), f"{100 * correct_count / test_count:.2f}%"],
            ["Predictions correct", "Predictions wrong", "Accuracy on this test"],
        ):
            with column:
                st.markdown(f'<div class="little-stat"><div class="n">{big}</div><div class="caption">{small}</div></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="safety"><b>Important:</b> The model missed <b>{misses} malignant samples</b> '
            'in this small test. Good benchmark accuracy does not prove that it is safe for patients.</div>',
            unsafe_allow_html=True,
        )
        st.write("**What does 97.37% accuracy mean?** It answered 111 of 114 test samples correctly. It does not mean it can diagnose someone with 97.37% certainty.")
        st.markdown("#### More technical charts (optional)")
        for label, file, caption in [
            ("📈 ROC curve, precision–recall and confusion matrix", "model_evaluation.png", "Three ways to examine performance on the 114 final test samples."),
            ("🔎 Which measurements matter most?", "feature_importance.png", "Permutation importance describes predictive influence; it does not establish medical causation."),
            ("📊 Model score calibration", "calibration.png", "Internal calibration analysis on a small test dataset."),
        ]:
            image_path = ROOT / "reports" / "figures" / file
            if image_path.is_file():
                with st.expander(label):
                    st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.warning("The saved report is missing. Run `python -m src.train` to recreate the report.")

st.divider()
st.markdown("""
**Where did these samples come from?** The public [Wisconsin Diagnostic Breast Cancer dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic),
available through scikit-learn. These are historical cell-measurement records, **not** live patients or personal medical files.

**See the code:** [Oncology ML Lab on GitHub](https://github.com/yashwantchatti005/oncology-ml-lab) ·
**Reminder:** educational research demonstration only, not for real medical decisions.
""")
