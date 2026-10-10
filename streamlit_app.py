"""IMDb movie-review sentiment classifier - Streamlit app.
Run:  pip install -r requirements.txt   then   streamlit run streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sentiment_core as core

CFG = core.CFG

st.set_page_config(page_title="IMDb Review Sentiment", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading the model (first run only)...")
def get_model():
    return core.load_model()


EXAMPLES = {
    "Clearly positive": "A moving, beautifully acted film with a script that kept me hooked from start to finish. "
                        "The ending was perfect and I would happily watch it again.",
    "Clearly negative": "The plot made no sense, the acting was wooden and I kept checking my watch. "
                        "A complete waste of two hours and the worst film I have seen this year.",
    "Mixed review": "The first half is slow and the dialogue is clumsy, but the performances are strong and "
                    "the final twenty minutes are genuinely gripping.",
}


def use_example(text):
    st.session_state["review"] = text
    st.session_state.pop("result", None)


with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"**Architecture:** {CFG['architecture']} "
        f"(128-dim embedding, 64 units, "
        f"embedding dropout {CFG.get('embedding_dropout', 0.4):.1f}, "
        f"dense dropout {CFG.get('dense_dropout', 0.5):.1f})"
    )
    st.metric("Expected accuracy (IMDb test reviews)", f"{CFG['reported_mean_test_accuracy']:.1%}",
              help=f"Mean of 3 training seeds, std {CFG['reported_std']:.1%}. "
                   f"This saved model scored {CFG['test_accuracy_this_model']:.1%}.")
    st.markdown(
        "**Limits**\n"
        f"- Reads only the last {core.MAXLEN} tokens of a review.\n"
        "- Words outside its 10,000-word vocabulary are treated as unknown.\n"
        "- Trained on IMDb reviews only. Sarcasm, double negatives and mixed reviews are often misread.\n"
        "- Probabilities near 0.5 are a toss-up."
    )

st.title("🎬 IMDb movie-review sentiment")
st.caption("Paste a movie review. The model classifies it as Positive or Negative and shows its probability.")

cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=use_example, args=(text,), key=f"example_{name}")

review = st.text_area("Movie review", key="review", height=200,
                      placeholder="Type or paste a review here...")

if st.button("Classify review", type="primary"):
    if not review.strip():
        st.session_state.pop("result", None)
        st.warning("Please type or paste a review first.")
    else:
        with st.spinner("Reading the review..."):
            result = core.classify(review, get_model())
        if result["n_words"] == 0:
            st.session_state.pop("result", None)
            st.warning("No readable words were found in that text.")
        else:
            st.session_state["result"] = result

result = st.session_state.get("result")
if result:
    p = min(max(result["p_positive"], 0.0), 1.0)
    banner = st.success if result["label"] == "Positive" else st.error
    banner(f"**{result['label'].upper()}** with {result['confidence']:.1%} confidence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", result["label"])
    c2.metric("Confidence", f"{result['confidence']:.1%}")
    c3.metric("P(Positive)", f"{p:.3f}")
    st.progress(p, text=f"Positive  {p:.1%}")
    st.progress(1 - p, text=f"Negative  {1 - p:.1%}")
    for note in result["notes"]:
        st.warning(note)
    with st.expander("What the model actually read"):
        st.write(f"{result['n_words']} words found. Words outside the vocabulary appear as <UNK>.")
        st.code(result["read"], language=None)