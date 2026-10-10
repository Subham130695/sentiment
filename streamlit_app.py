"""IMDb movie-review sentiment classifier - Streamlit app (white & orange theme).

Run: pip install -r requirements.txt then streamlit run streamlit_app.py"""

import html
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sentiment_core as core

CFG = core.CFG

st.set_page_config(page_title="IMDb Review Sentiment", page_icon="🎬", layout="centered")

# ---------------------------------------------------------------------------
# Theme: white + orange
# (base colours also live in .streamlit/config.toml; this CSS handles the rest)
# ---------------------------------------------------------------------------
ORANGE = "#F97316"
ORANGE_DARK = "#EA580C"
ORANGE_SOFT = "#FFF4EA"
ORANGE_LINE = "#FDBA74"
INK = "#1F2937"
MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
    /* page + sidebar */
    .stApp {{ background: #FFFFFF; color: {INK}; }}
    [data-testid="stSidebar"] {{
        background: {ORANGE_SOFT};
        border-right: 1px solid {ORANGE_LINE};
    }}
    [data-testid="stSidebar"] h2 {{ color: {ORANGE_DARK}; }}
    [data-testid="stHeader"] {{ background: transparent; }}

    /* thin orange bar across the top */
    .stApp::before {{
        content: "";
        position: fixed; top: 0; left: 0; right: 0; height: 4px;
        background: {ORANGE}; z-index: 999991;
    }}

    /* title + caption */
    h1 {{ color: {INK}; font-weight: 800; letter-spacing: -0.5px; }}
    [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

    /* metric values */
    [data-testid="stMetricValue"] {{ color: {ORANGE_DARK}; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

    /* example buttons (secondary) */
    .stButton > button {{
        background: #FFFFFF;
        color: {ORANGE_DARK};
        border: 1px solid {ORANGE_LINE};
        border-radius: 8px;
        transition: all .15s ease;
    }}
    .stButton > button:hover {{
        background: {ORANGE_SOFT};
        border-color: {ORANGE};
        color: {ORANGE_DARK};
    }}

    /* primary button (Classify review) */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {ORANGE};
        color: #FFFFFF;
        border: 1px solid {ORANGE};
        font-weight: 600;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {ORANGE_DARK};
        border-color: {ORANGE_DARK};
        color: #FFFFFF;
    }}

    /* text area */
    div[data-baseweb="textarea"] {{
        background: #FFFFFF;
        border: 1px solid {ORANGE_LINE};
        border-radius: 10px;
    }}
    div[data-baseweb="textarea"] textarea {{ background: #FFFFFF; color: {INK}; }}
    div[data-baseweb="textarea"]:focus-within {{
        border-color: {ORANGE};
        box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.18);
    }}

    /* progress bars: orange fill on a light track */
    [data-testid="stProgress"] > div > div > div {{ background-color: #F3F4F6; }}
    [data-testid="stProgress"] > div > div > div > div {{ background-color: {ORANGE}; }}

    /* expander + code block */
    [data-testid="stExpander"] {{
        border: 1px solid {ORANGE_LINE};
        border-radius: 10px;
        background: #FFFFFF;
    }}
    [data-testid="stExpander"] summary:hover {{ color: {ORANGE_DARK}; }}
    [data-testid="stCode"], pre {{ background: {ORANGE_SOFT} !important; }}

    /* result banner */
    .result-banner {{
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 0.98rem;
        margin: 6px 0 14px 0;
    }}
    .result-banner.pos {{
        background: {ORANGE_SOFT};
        border: 1px solid {ORANGE};
        border-left: 6px solid {ORANGE};
        color: #9A3412;
    }}
    .result-banner.neg {{
        background: #F3F4F6;
        border: 1px solid #D1D5DB;
        border-left: 6px solid {INK};
        color: {INK};
    }}

    /* notes / warnings */
    .note-box {{
        padding: 12px 16px;
        border-radius: 10px;
        background: #FFFBEB;
        border: 1px solid #FCD34D;
        color: #78350F;
        font-size: 0.92rem;
        margin: 10px 0;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def show_banner(label, confidence):
    css_class = "pos" if label == "Positive" else "neg"
    st.markdown(
        f'<div class="result-banner {css_class}"><strong>{html.escape(label.upper())}</strong> '
        f"with {confidence:.1%} confidence</div>",
        unsafe_allow_html=True,
    )


def show_note(text):
    st.markdown(f'<div class="note-box">{html.escape(str(text))}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"**Architecture:** {CFG['architecture']} "
        f"(128-dim embedding, 64 units, "
        f"embedding dropout {CFG.get('embedding_dropout', 0.4):.1f}, "
        f"dense dropout {CFG.get('dense_dropout', 0.5):.1f})"
    )
    st.metric(
        "Expected accuracy (IMDb test reviews)",
        f"{CFG['reported_mean_test_accuracy']:.1%}",
        help=f"Mean of 3 training seeds, std {CFG['reported_std']:.1%}. "
             f"This saved model scored {CFG['test_accuracy_this_model']:.1%}.",
    )
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

    show_banner(result["label"], result["confidence"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", result["label"])
    c2.metric("Confidence", f"{result['confidence']:.1%}")
    c3.metric("P(Positive)", f"{p:.3f}")

    st.progress(p, text=f"Positive {p:.1%}")
    st.progress(1 - p, text=f"Negative {1 - p:.1%}")

    for note in result["notes"]:
        show_note(note)

    with st.expander("What the model actually read"):
        st.write(f"{result['n_words']} words found. Words outside the vocabulary appear as <UNK>.")
        st.code(result["read"], language=None)
