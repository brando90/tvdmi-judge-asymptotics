"""Pairwise judge features shared by the three MT-Bench evaluations."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def mtbench_features(df: pd.DataFrame, train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute non-human pairwise judge signals from real GPT-4 votes + conversation texts."""
    prompts, resp_a, resp_b = [], [], []
    for _, row in df.iterrows():
        turn_idx = 0 if int(row["turn"]) == 1 else 1
        ca, cb = row["conversation_a"], row["conversation_b"]
        u_txt = str(ca[2 * turn_idx]["content"])
        a_txt = str(ca[2 * turn_idx + 1]["content"])
        b_txt = str(cb[2 * turn_idx + 1]["content"])
        prompts.append(u_txt)
        resp_a.append(a_txt)
        resp_b.append(b_txt)

    # TF-IDF prompt-response relevance difference (A minus B)
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    vec.fit([texts[i] for texts in (prompts, resp_a, resp_b) for i in np.flatnonzero(train)])
    VP = vec.transform(prompts)
    VA = vec.transform(resp_a)
    VB = vec.transform(resp_b)
    sim_a = np.asarray(VP.multiply(VA).sum(axis=1)).ravel()
    sim_b = np.asarray(VP.multiply(VB).sum(axis=1)).ravel()
    tfidf_diff = sim_a - sim_b

    # Unique non-stopword lexical richness difference
    def lex_div(txt: str) -> float:
        toks = re.findall(r"[a-z]+", txt.lower())
        return len(set(toks)) / np.sqrt(len(toks) + 1.0)

    lex_diff = np.array([lex_div(a) - lex_div(b) for a, b in zip(resp_a, resp_b)])
    # Character length and word length diffs (verbosity clones)
    wlen_diff = np.array([len(a.split()) - len(b.split()) for a, b in zip(resp_a, resp_b)], dtype=np.float64)
    clen_diff = np.array([len(a) - len(b) for a, b in zip(resp_a, resp_b)], dtype=np.float64)

    # GPT-4 real pairwise vote mapped to +1 (model_a), -1 (model_b), 0 (tie)
    gmap = {"model_a": 1.0, "model_b": -1.0, "tie": 0.0, "tie (inconsistent)": 0.0}
    gpt4_score = np.array([gmap[w] for w in df["winner_g"]], dtype=np.float64)

    rng_mt = np.random.default_rng(seed + 99)
    # Construct K=17 realistic messy pairwise judge pool:
    # 1 real GPT-4, 1 TF-IDF prompt-alignment, 1 lexical richness,
    # 11 verbosity/length clones (strong confounding clone family!), 3 sign-flipped adversaries!
    cols = [
        gpt4_score,
        tfidf_diff,
        lex_diff,
    ]
    for c_idx in range(11):
        base = wlen_diff if c_idx % 2 == 0 else clen_diff
        jitter = rng_mt.normal(0.0, 0.05 * base[train].std(), size=len(base))
        cols.append(base + jitter)
    cols.extend([-gpt4_score, -tfidf_diff, -lex_diff])
    return np.column_stack(cols), wlen_diff
