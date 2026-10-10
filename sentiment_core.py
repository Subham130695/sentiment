"""Text -> model input and prediction logic for the IMDb sentiment app (no Streamlit dependency)."""
import json
import os
import re
import unicodedata

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "deploy_config.json")))
WORD_INDEX = json.load(open(os.path.join(HERE, "word_index.json")))   # word -> rank (1 = most common)
MAXLEN = int(CFG["maxlen"])

ID_TO_WORD = {rank + 3: word for word, rank in WORD_INDEX.items()}
ID_TO_WORD.update({1: "<START>", 2: "<UNK>"})

_PIECE = re.compile(r"[\w']+|[^\w\s']")     # a word (apostrophes kept) or one stray symbol
_WORD = re.compile(r"[\w']+")


def tokenize(text):
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return _PIECE.findall(text)


def to_id(tok):
    """Integer id for one token, or None if the token is dropped (punctuation)."""
    rank = WORD_INDEX.get(tok)
    if rank is not None:                       # known word (rank + 3, as in the Keras IMDb data)
        return rank + 3
    if _WORD.fullmatch(tok):                   # an unknown word
        core = tok.strip("'")
        if not core:
            return None                        # only apostrophes
        rank = WORD_INDEX.get(core) if core != tok else None
        return rank + 3 if rank is not None else 2
    # a single stray symbol: ordinary punctuation is dropped, anything else is an unknown token
    if ord(tok) < 128 or unicodedata.category(tok).startswith("P"):
        return None
    return 2


def encode(text):
    """Same preprocessing as training: START token, word ids, pre-padding and pre-truncation to MAXLEN."""
    ids = [1] + [i for i in map(to_id, tokenize(text)) if i is not None]
    used = ids[-MAXLEN:]
    x = np.zeros(MAXLEN, dtype="int32")
    x[-len(used):] = used
    info = {"n_words": len(ids) - 1, "n_ids": len(ids), "truncated": len(ids) > MAXLEN,
            "unk_share": float(np.mean(np.array(used) == 2))}
    return x, info


def decode(x):
    """The text the model actually read (padding removed)."""
    return " ".join(ID_TO_WORD.get(int(t), "<UNK>") for t in x if t != 0)


def load_model():
    from tensorflow import keras
    return keras.models.load_model(os.path.join(HERE, "BEST_model.keras"), compile=False)


def predict_proba(model, x):
    out = model(x[None, :], training=False)
    out = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
    return float(out[0, 0])


def classify(text, model):
    x, info = encode(text)
    p = predict_proba(model, x)
    half = float(CFG["uncertain_half_width"])
    rob = CFG["robustness_accuracy"]
    notes = []
    if abs(p - 0.5) < half:
        msg = f"Low confidence: the probability is between {0.5 - half:.2f} and {0.5 + half:.2f}."
        if CFG.get("low_conf_accuracy") is not None:
            msg += (f" In testing, predictions in this range were right only about "
                    f"{CFG['low_conf_accuracy']:.0%} of the time, close to a coin flip.")
        notes.append(msg)
    if info["truncated"]:
        notes.append(f"Only the last {MAXLEN} tokens were analysed (this review has {info['n_words']} words), so the "
                     "beginning was ignored. Longer reviews were classified less accurately in testing.")
    if info["n_ids"] < CFG["warn_short_tokens"]:
        notes.append("Very short review: it is shorter than 95% of the reviews the model was trained on, "
                     "so the model has seen few like it.")
    if info["unk_share"] > CFG["warn_unk_share"]:
        drop = (rob["clean"] - rob["UNK 20%"]) * 100
        notes.append(f"Many unknown words ({info['unk_share']:.0%} are outside the model's 10,000-word vocabulary). "
                     f"In testing, hiding 20% of the words cost this model about {drop:.1f} points of accuracy.")
    return {"label": "Positive" if p >= 0.5 else "Negative", "p_positive": p,
            "confidence": max(p, 1 - p), "notes": notes, "read": decode(x), **info}