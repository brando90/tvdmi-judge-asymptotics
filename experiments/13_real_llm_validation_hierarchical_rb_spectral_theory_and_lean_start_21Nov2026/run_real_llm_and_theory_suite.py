"""Experiment 13 — Real-LLM Validation (SummEval & MT-Bench), Hierarchical RB-Spec TVD-MI, and Numerical Theory Verification.

Usage:
    python3 run_real_llm_and_theory_suite.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "text.usetex": False,
    }
)

EXP13_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP13_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMEVAL_CACHE = Path("/tmp/corr-ce-public-ordinal/summeval")
MTBENCH_CACHE = Path("/tmp/mt_bench_public")

SEED = 42
N_FOLDS = 5
N_SPLITS_ROBUST = 200
N_BOOT = 2000
N_MC = 500
N_PERM = 500

GRADE_LINE = re.compile(r"\A\s*([1-5])(?:[ \t]*/[ \t]*5)?[ \t]*(?:\r?\n|\Z)")


class NumericUnpickler(pickle.Unpickler):
    """Restricted unpickler matching the SHA-256-verified BARTScore SummEval archive."""

    def find_class(self, module: str, name: str):
        from numpy.core.multiarray import scalar

        allowed = {
            ("numpy.core.multiarray", "scalar"): scalar,
            ("numpy", "dtype"): np.dtype,
        }
        return allowed[(module, name)]


def icc_a1_and_c1(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute Shrout-Fleiss two-way random single-measure ICC(A,1) and ICC(C,1)."""
    n = len(pred)
    if n < 3:
        return float("nan"), float("nan")
    M = np.column_stack([target, pred]).astype(np.float64)
    k = 2.0
    row_mean = M.mean(axis=1)
    col_mean = M.mean(axis=0)
    grand = M.mean()
    msb = k * np.sum((row_mean - grand) ** 2) / (n - 1)
    msj = n * np.sum((col_mean - grand) ** 2) / (k - 1)
    resid = M - row_mean[:, None] - col_mean[None, :] + grand
    mse = np.sum(resid ** 2) / ((n - 1) * (k - 1))
    denom_a = msb + (k - 1) * mse + k * (msj - mse) / n
    denom_c = msb + (k - 1) * mse
    icc_a1 = float((msb - mse) / denom_a) if abs(denom_a) > 1e-12 else 0.0
    icc_c1 = float((msb - mse) / denom_c) if abs(denom_c) > 1e-12 else 0.0
    return icc_a1, icc_c1


def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    """Return correlation statistic safely when arrays may be near-constant."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    val = float(fn(a, b).statistic)
    return 0.0 if np.isnan(val) else val


def load_summeval_bundle() -> dict:
    """Load SHA-256 verified SummEval annotations, GPT-4 completions, and BART/PRISM metrics."""
    ann_path = SUMMEVAL_CACHE / "model_annotations.aligned.jsonl"
    bart_path = SUMMEVAL_CACHE / "bart_metrics.pkl"
    annotations = [json.loads(line) for line in ann_path.read_bytes().splitlines()]
    raw = {(r["id"], r["model_id"]): r for r in annotations}
    docs = sorted({a for a, _ in raw})
    systems = sorted({s for _, s in raw})
    assert len(docs) == 100 and len(systems) == 16 and len(raw) == 1600

    with bart_path.open("rb") as f:
        bart = NumericUnpickler(f).load()

    dims = ["consistency", "relevance", "coherence", "fluency"]
    experts = np.zeros((100, 16, 3, 4), dtype=np.float64)
    for di, doc in enumerate(docs):
        for si, sys_id in enumerate(systems):
            for ei, ann in enumerate(raw[(doc, sys_id)]["expert_annotations"]):
                for ki, dim in enumerate(dims):
                    experts[di, si, ei, ki] = float(ann[dim])

    H_item_dim = experts.mean(axis=2)  # (100, 16, 4)
    H_overall = H_item_dim.mean(axis=2)  # (100, 16)

    geval_files = {
        "con": "gpt4_con_detailed.json",
        "rel": "gpt4_rel_detailed.json",
        "coh": "gpt4_coh_detailed.json",
        "flu": "gpt4_flu_detailed.json",
    }
    G_dims = {}
    parse_counts = {}
    for short, fn in geval_files.items():
        recs = {(r["doc_id"], r["system_id"]): r for r in json.loads((SUMMEVAL_CACHE / fn).read_bytes())}
        M = np.full((100, 16, 20), np.nan, dtype=np.float64)
        valid = 0
        total = 0
        for di, doc in enumerate(docs):
            for si, sys_id in enumerate(systems):
                resps = recs[(doc, sys_id)]["all_responses"][:20]
                for j, txt in enumerate(resps):
                    total += 1
                    m = GRADE_LINE.match(str(txt))
                    if m:
                        M[di, si, j] = float(m.group(1))
                        valid += 1
                row = M[di, si]
                if np.isnan(row).any():
                    row[np.isnan(row)] = np.nanmean(row) if not np.isnan(row).all() else 3.0
        G_dims[short] = M
        parse_counts[short] = {"valid": valid, "total": total}

    s0 = bart[docs[0]]["sys_summs"][systems[0]]["scores"]
    en_keys = sorted([k for k in s0 if k.startswith("bart_score_cnn_src_hypo_en_")])
    rf_keys = [
        "bart_score_src_hypo",
        "bart_score_para_src_hypo",
        "bart_score_cnn_src_hypo",
        "prism_src_hypo",
    ]
    B_en = np.array(
        [[[float(bart[d]["sys_summs"][s]["scores"][k]) for k in en_keys] for s in systems] for d in docs],
        dtype=np.float64,
    )
    B_rf = np.array(
        [[[float(bart[d]["sys_summs"][s]["scores"][k]) for k in rf_keys] for s in systems] for d in docs],
        dtype=np.float64,
    )
    X_docs = [bart[d]["src"] for d in docs]
    Y_cand = [[bart[d]["sys_summs"][s]["sys_summ"] for s in systems] for d in docs]
    word_len = np.array([[len(Y_cand[di][si].split()) for si in range(16)] for di in range(100)], dtype=np.float64)

    # Trace 1 exact example byte-for-byte (verify zero human score leakage into judge inputs)
    d0, s0_id = docs[0], systems[0]
    con_rec0 = {(r["doc_id"], r["system_id"]): r for r in json.loads((SUMMEVAL_CACHE / "gpt4_con_detailed.json").read_bytes())}[(d0, s0_id)]
    trace_item = {
        "article_id": d0,
        "system_id": s0_id,
        "article_sha256": hashlib.sha256(bart[d0]["src"].encode("utf-8")).hexdigest(),
        "candidate_sha256": hashlib.sha256(bart[d0]["sys_summs"][s0_id]["sys_summ"].encode("utf-8")).hexdigest(),
        "human_keywords_in_gpt4_prompt": any(k in con_rec0["prompt"].lower() for k in ("expert_annotation", "turker_annotation")),
        "expert_scores_4dims": H_item_dim[0, 0].tolist(),
        "gpt4_con_20_parsed": G_dims["con"][0, 0].tolist(),
        "rf_seq2seq_scores": B_rf[0, 0].tolist(),
        "candidate_word_count": int(word_len[0, 0]),
    }
    (RESULTS_DIR / "trace_one_item.json").write_text(json.dumps(trace_item, indent=2), encoding="utf-8")

    return {
        "docs": docs,
        "systems": systems,
        "experts": experts,
        "H_item_dim": H_item_dim,
        "H_overall": H_overall,
        "G_dims": G_dims,
        "parse_counts": parse_counts,
        "B_en": B_en,
        "B_rf": B_rf,
        "X_docs": X_docs,
        "Y_cand": Y_cand,
        "word_len": word_len,
        "trace_item": trace_item,
    }


def compute_human_ceiling(bundle: dict, rng: np.random.Generator) -> dict:
    """Compute leave-one-expert-out ceiling across the 3 SummEval experts on 1,600 summaries."""
    exp_overall = bundle["experts"].mean(axis=3)  # (100, 16, 3)
    item_metrics = {m: [] for m in ("spearman", "kendall", "pearson", "icc_a1", "icc_c1", "mae", "bias")}
    sys_sp = []
    # For independent article-cluster bootstrap, average the 3 leave-one-out pairings per article
    art_sp = np.zeros((100, 3))
    art_bias = np.zeros((100, 3))
    for rater in range(3):
        others = [j for j in range(3) if j != rater]
        pred = exp_overall[:, :, rater]
        ref = exp_overall[:, :, others].mean(axis=2)
        p_flat, r_flat = pred.ravel(), ref.ravel()
        item_metrics["spearman"].append(safe_corr(spearmanr, p_flat, r_flat))
        item_metrics["kendall"].append(safe_corr(kendalltau, p_flat, r_flat))
        item_metrics["pearson"].append(safe_corr(pearsonr, p_flat, r_flat))
        ia, ic = icc_a1_and_c1(p_flat, r_flat)
        item_metrics["icc_a1"].append(ia)
        item_metrics["icc_c1"].append(ic)
        item_metrics["mae"].append(float(np.mean(np.abs(p_flat - r_flat))))
        item_metrics["bias"].append(float(np.mean(p_flat - r_flat)))
        sys_sp.append(safe_corr(spearmanr, pred.mean(0), ref.mean(0)))
        for a in range(100):
            art_sp[a, rater] = safe_corr(spearmanr, pred[a], ref[a])
            art_bias[a, rater] = float(np.mean(pred[a] - ref[a]))

    # Bootstrap over the 100 articles for uncertainty on leave-one-expert-out ceiling
    boot_sp = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, 100, size=100)
        vals = []
        for rater in range(3):
            others = [j for j in range(3) if j != rater]
            vals.append(safe_corr(spearmanr, exp_overall[idx, :, rater].ravel(), exp_overall[idx][:, :, others].mean(2).ravel()))
        boot_sp.append(np.mean(vals))

    return {
        "n_articles": 100,
        "n_items": 1600,
        "mean_item_spearman": float(np.mean(item_metrics["spearman"])),
        "ci_item_spearman": [float(np.quantile(boot_sp, 0.025)), float(np.quantile(boot_sp, 0.975))],
        "mean_item_kendall": float(np.mean(item_metrics["kendall"])),
        "mean_item_pearson": float(np.mean(item_metrics["pearson"])),
        "mean_item_icc_a1": float(np.mean(item_metrics["icc_a1"])),
        "mean_item_icc_c1": float(np.mean(item_metrics["icc_c1"])),
        "mean_item_mae": float(np.mean(item_metrics["mae"])),
        "mean_item_bias": float(np.mean(item_metrics["bias"])),
        "mean_sys_spearman": float(np.mean(sys_sp)),
        "mean_within_article_spearman": float(art_sp.mean()),
        "human_overall_mean": float(exp_overall.mean()),
    }


def module2_falsification_and_double_centering(bundle: dict) -> dict:
    """Test prompt-copy blindness, cross-article swap, and additive shift invariance on SummEval texts."""
    X_docs = bundle["X_docs"]
    Y_cand = bundle["Y_cand"]
    H_sys = bundle["H_overall"].mean(axis=0)
    H_con = bundle["H_item_dim"][:, :, 0].mean(axis=0)

    lead3_docs = [" ".join(re.split(r"(?<=[.!?])\s+", x.strip())[:3]) for x in X_docs]
    all_texts = X_docs + lead3_docs + [Y_cand[di][si] for di in range(100) for si in range(16)]
    vec = TfidfVectorizer(max_features=4000, stop_words="english")
    vec.fit(all_texts)
    VX = vec.transform(X_docs).toarray()
    V_lead3 = vec.transform(lead3_docs).toarray()
    VY = np.array([vec.transform([Y_cand[di][si] for si in range(16)]).toarray() for di in range(100)])

    def tvd_witness_pair(U: np.ndarray, V: np.ndarray) -> float:
        S = U @ V.T
        n = U.shape[0]
        joint = np.diag(S).mean()
        prod = (S.sum() - np.diag(S).sum()) / (n * (n - 1))
        return float(0.5 * (joint - prod))

    def cross_peer_tvd(target_vec: np.ndarray, omit_si: int | None = None) -> float:
        vals = [tvd_witness_pair(target_vec, VY[:, pj, :]) for pj in range(16) if pj != omit_si]
        return float(np.mean(vals))

    pr_sys = np.array([tvd_witness_pair(VX, VY[:, si, :]) for si in range(16)])
    peer_sys = np.array([cross_peer_tvd(VY[:, si, :], omit_si=si) for si in range(16)])

    copy_pr = tvd_witness_pair(VX, VX)
    copy_peer = cross_peer_tvd(VX)
    lead3_pr = tvd_witness_pair(VX, V_lead3)
    lead3_peer = cross_peer_tvd(V_lead3)
    swap_idx = np.roll(np.arange(100), 17)
    swap_pr = tvd_witness_pair(VX, VY[swap_idx, int(np.argmax(H_sys)), :])

    # Additive shift invariance check on real item scores (GPT4-con + additive prompt shift + len shift)
    G_mat = bundle["G_dims"]["con"][:, :, 0]  # (100, 16)
    rng_shift = np.random.default_rng(SEED)
    a_prompt = rng_shift.normal(0.8, 0.4, size=(100, 1))
    b_sys = rng_shift.normal(-0.5, 0.3, size=(1, 16))
    G_shifted = G_mat + a_prompt + b_sys

    def double_center(M: np.ndarray) -> np.ndarray:
        return M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()

    dc_max_diff = float(np.max(np.abs(double_center(G_shifted) - double_center(G_mat))))
    raw_max_diff = float(np.max(np.abs(G_shifted - G_mat)))
    pc_max_diff = float(
        np.max(np.abs((G_shifted - G_shifted.mean(1, keepdims=True)) - (G_mat - G_mat.mean(1, keepdims=True))))
    )

    out = {
        "prompt_tvd_sys_spearman_overall": safe_corr(spearmanr, pr_sys, H_sys),
        "prompt_tvd_sys_spearman_consistency": safe_corr(spearmanr, pr_sys, H_con),
        "peer_tvd_sys_spearman_overall": safe_corr(spearmanr, peer_sys, H_sys),
        "peer_tvd_sys_spearman_consistency": safe_corr(spearmanr, peer_sys, H_con),
        "top_real_sys_prompt_tvd": {"system": bundle["systems"][int(np.argmax(pr_sys))], "score": float(pr_sys.max())},
        "max_real_prompt_tvd": float(pr_sys.max()),
        "verbatim_copy_prompt_tvd": copy_pr,
        "lead3_extractive_prompt_tvd": lead3_pr,
        "cross_article_swap_prompt_tvd": swap_pr,
        "verbatim_copy_peer_tvd": copy_peer,
        "lead3_extractive_peer_tvd": lead3_peer,
        "top_real_peer_tvd": float(peer_sys.max()),
        "additive_shift_raw_max_delta": raw_max_diff,
        "additive_shift_prompt_centered_only_delta": pc_max_diff,
        "additive_shift_double_centered_delta": dc_max_diff,
    }
    return out


def avg_pairwise_error_corr(scores_2d: np.ndarray, target_1d: np.ndarray) -> float:
    """Compute mean off-diagonal correlation among standardized judge error residuals."""
    Z = (scores_2d - scores_2d.mean(axis=0)) / (scores_2d.std(axis=0) + 1e-12)
    t = (target_1d - target_1d.mean()) / (target_1d.std() + 1e-12)
    E = Z - t[:, None]
    C = np.corrcoef(E, rowvar=False)
    K = C.shape[0]
    if K < 2:
        return 0.0
    return float((C.sum() - np.trace(C)) / (K * (K - 1)))


def module3_saturation_and_split_null(bundle: dict, rng: np.random.Generator) -> dict:
    """Measure within-family saturation vs cross-family gain and article-permutation null bias."""
    H_flat = bundle["H_overall"].ravel()
    H_con_flat = bundle["H_item_dim"][:, :, 0].ravel()

    G_con = bundle["G_dims"]["con"].reshape(1600, 20)
    G_rel = bundle["G_dims"]["rel"].reshape(1600, 20)
    B_en = bundle["B_en"].reshape(1600, 70)
    B_rf = bundle["B_rf"].reshape(1600, 4)

    rho_gpt4_con = avg_pairwise_error_corr(G_con, H_con_flat)
    rho_gpt4_rel = avg_pairwise_error_corr(G_rel, bundle["H_item_dim"][:, :, 1].ravel())
    rho_bart_70 = avg_pairwise_error_corr(B_en, H_flat)
    cross5 = np.column_stack([G_con[:, 0], B_rf[:, 0], B_rf[:, 1], B_rf[:, 2], B_rf[:, 3]])
    rho_cross5 = avg_pairwise_error_corr(cross5, H_flat)

    # K-scaling within-family (GPT4-con 1..20 & BART-en 1..70) vs F-scaling (1..5 distinct models)
    within_gpt_curve = []
    for k in (1, 2, 4, 8, 12, 16, 20):
        sc = G_con[:, :k].mean(axis=1)
        within_gpt_curve.append({"K": k, "item_spearman": safe_corr(spearmanr, sc, H_con_flat)})

    # Permutation null bias on real SummEval (permute article order for candidate-article link)
    null_pool = np.column_stack([
        bundle["G_dims"]["con"].reshape(1600, 20),
        bundle["G_dims"]["rel"].reshape(1600, 20),
        bundle["B_en"][:, :, :40].reshape(1600, 40),
    ])  # 80 real critics
    null_pool_std = (null_pool - null_pool.mean(0)) / (null_pool.std(0) + 1e-12)
    null_pool_3d = null_pool_std.reshape(100, 16, 80)
    null_rows = []
    for n_art in (20, 50):
        half = n_art // 2
        for k_sub in (1, 4, 16, 40, 80):
            naive_vals = np.empty(N_PERM)
            split_vals = np.empty(N_PERM)
            for r in range(N_PERM):
                art_idx = rng.choice(100, size=n_art, replace=False)
                # Under strict null, permute all flattened (article, system) pairings
                # so neither article difficulty nor system quality leaks across candidates.
                sub_X = null_pool_3d[art_idx, :, :k_sub]  # (n_art, 16, k_sub)
                h_flat_sub = bundle["H_overall"][art_idx, :].ravel()
                h_perm_flat = h_flat_sub[rng.permutation(len(h_flat_sub))]
                n_half_items = half * 16
                sel_X = sub_X[:half].reshape(-1, k_sub)
                eval_X = sub_X[half:].reshape(-1, k_sub)
                h_sel = (h_perm_flat[:n_half_items] - h_perm_flat[:n_half_items].mean()) / (
                    h_perm_flat[:n_half_items].std() + 1e-12
                )
                h_ev = (h_perm_flat[n_half_items:] - h_perm_flat[n_half_items:].mean()) / (
                    h_perm_flat[n_half_items:].std() + 1e-12
                )
                h_all = (h_perm_flat - h_perm_flat.mean()) / (h_perm_flat.std() + 1e-12)
                scores_sel = (sel_X.T @ h_sel) / len(h_sel)
                scores_ev = (eval_X.T @ h_ev) / len(h_ev)
                scores_full = (sub_X.reshape(-1, k_sub).T @ h_all) / len(h_all)
                naive_vals[r] = float(np.max(scores_full))
                j_star = int(np.argmax(scores_sel))
                split_vals[r] = float(scores_ev[j_star])
            w_p = float(wilcoxon(np.abs(naive_vals), np.abs(split_vals)).pvalue) if k_sub > 1 else float("nan")
            sp_zero_p = float(wilcoxon(split_vals).pvalue)
            null_rows.append({
                "n_art": n_art,
                "K": k_sub,
                "naive_mean": float(naive_vals.mean()),
                "naive_ci": [float(np.quantile(naive_vals, 0.025)), float(np.quantile(naive_vals, 0.975))],
                "split_mean": float(split_vals.mean()),
                "split_ci": [float(np.quantile(split_vals, 0.025)), float(np.quantile(split_vals, 0.975))],
                "split_zero_bias_p": sp_zero_p,
                "wilcoxon_abs_p": w_p,
            })

    return {
        "rho_within_gpt4_con_20": rho_gpt4_con,
        "rho_within_gpt4_rel_20": rho_gpt4_rel,
        "rho_within_bart_70": rho_bart_70,
        "rho_cross_family_5": rho_cross5,
        "within_gpt4_k_curve": within_gpt_curve,
        "null_overfitting_rows": null_rows,
    }


def _standardize_fold(J_tr: np.ndarray, J_te: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize 3D (n_docs, n_sys, K) judge arrays using training fold statistics only."""
    tr_flat = J_tr.reshape(-1, J_tr.shape[2])
    te_flat = J_te.reshape(-1, J_te.shape[2])
    mu = tr_flat.mean(axis=0)
    sd = np.maximum(tr_flat.std(axis=0), 1e-8)
    Z_tr = (tr_flat - mu) / sd
    Z_te = (te_flat - mu) / sd
    return Z_tr.reshape(J_tr.shape), Z_te.reshape(J_te.shape), mu, sd


def _fit_and_predict_aggregators(
    J_tr: np.ndarray, J_te: np.ndarray, anchor_idx: list[int]
) -> dict[str, np.ndarray]:
    """Fit all unsupervised aggregators strictly on training fold J_tr and predict [1,5] on J_te."""
    n_tr, n_sys, K = J_tr.shape
    n_te = J_te.shape[0]
    Z_tr3d, Z_te3d, _, _ = _standardize_fold(J_tr, J_te)
    Z_tr = Z_tr3d.reshape(-1, K)
    Z_te = Z_te3d.reshape(-1, K)

    # Anchor scale on [1,5] learned strictly from raw unsupervised training GPT-4 Likert items
    # (or neutral Likert prior (3.0, 0.8) if pool contains no [1,5]-Likert judge, e.g. Seq2Seq-only).
    if anchor_idx:
        raw_anchor = J_tr[:, :, anchor_idx]
        mu_anchor = float(raw_anchor.mean())
        sd_anchor = float(max(raw_anchor.std(), 1e-3))
    else:
        mu_anchor = 3.0
        sd_anchor = 0.8

    def scale_pred(raw_w: np.ndarray, eval_mat: np.ndarray, train_mat: np.ndarray) -> np.ndarray:
        tr_scores = train_mat @ raw_w
        tr_mu, tr_sd = tr_scores.mean(), max(tr_scores.std(), 1e-8)
        te_z = ((eval_mat @ raw_w) - tr_mu) / tr_sd
        return np.clip(mu_anchor + sd_anchor * te_z, 1.0, 5.0)

    # 1. Single GPT4-con judge (if in pool at col 0)
    pred_single = np.clip(J_te[:, :, 0].ravel(), 1.0, 5.0)

    # 2. Uniform Mean over [1,5]-rescaled judges (min-max fit on train only)
    tr_f = J_tr.reshape(-1, K)
    mn = tr_f.min(axis=0)
    mx = np.maximum(tr_f.max(axis=0), mn + 1e-8)
    U_te = 1.0 + 4.0 * np.clip((J_te.reshape(-1, K) - mn) / (mx - mn), 0.0, 1.0)
    pred_mean = U_te.mean(axis=1)
    pred_median = np.median(U_te, axis=1)

    # 3. Split-Sample Best-Spread Critic (largest within-article std on Z_tr3d)
    C_tr = (Z_tr3d - Z_tr3d.mean(axis=1, keepdims=True)).reshape(-1, K)
    spread = np.abs(C_tr).mean(axis=0)
    w_best = np.zeros(K)
    w_best[int(np.argmax(spread))] = 1.0
    pred_best = scale_pred(w_best, Z_te, Z_tr)

    # 4. Naive Rank-1 SVD (no prompt centering, no redundancy balancing)
    _, _, vt_naive = np.linalg.svd(Z_tr, full_matrices=False)
    v_naive = vt_naive[0]
    if np.sum(v_naive) < 0:
        v_naive = -v_naive
    pred_naive_svd = scale_pred(v_naive, Z_te, Z_tr)

    # 5. RB-Spec TVD-MI (within-prompt centered covariance -> redundancy mass m_j = sum |R|^6 -> balanced SVD)
    C_tr_std = C_tr / (np.maximum(C_tr.std(axis=0), 1e-8))
    R = np.corrcoef(C_tr_std, rowvar=False)
    if np.ndim(R) < 2:
        R = np.ones((1, 1))
    R = np.nan_to_num(R, nan=0.0)
    eff_mass = np.maximum(np.sum(np.abs(R) ** 6, axis=1), 1.0)
    bal_scale = eff_mass ** (-0.5)
    C_bal = C_tr_std * bal_scale[None, :]
    _, _, vt_rb = np.linalg.svd(C_bal, full_matrices=False)
    v_rb = vt_rb[0] * bal_scale
    # Redundancy-balanced sign orientation: sum(m_j^{-1/2} * sign(v_rb) * |v_rb|)
    # Each independent cluster has L1 mass ~ sqrt(K_f) * m_f^{-1/2} = O(1), invariant to clone floods!
    if np.sum(v_rb * bal_scale) < 0:
        v_rb = -v_rb
    pred_rb_spec = scale_pred(v_rb, Z_te, Z_tr)

    # Normalize test predictions into (n_te, n_sys)
    return {
        "Single GPT4-con": pred_single.reshape(n_te, n_sys),
        "Best-Spread Split": pred_best.reshape(n_te, n_sys),
        "Uniform Mean": pred_mean.reshape(n_te, n_sys),
        "Itemwise Median": pred_median.reshape(n_te, n_sys),
        "Naive SVD": pred_naive_svd.reshape(n_te, n_sys),
        "RB-Spec TVD-MI": pred_rb_spec.reshape(n_te, n_sys),
    }


def eval_5fold_summeval(
    J_full: np.ndarray,
    H_target: np.ndarray,
    anchor_idx: list[int],
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict:
    """Evaluate all aggregators on 5-fold article-disjoint CV with article-cluster bootstrap."""
    art_perm = rng.permutation(100)
    folds = np.array_split(art_perm, N_FOLDS)
    method_names = ["Single GPT4-con", "Best-Spread Split", "Uniform Mean", "Itemwise Median", "Naive SVD", "RB-Spec TVD-MI"]
    oof_preds = {m: np.zeros((100, 16), dtype=np.float64) for m in method_names}

    for test_idx in folds:
        mask = np.ones(100, dtype=bool)
        mask[test_idx] = False
        train_idx = np.where(mask)[0]
        preds = _fit_and_predict_aggregators(J_full[train_idx], J_full[test_idx], anchor_idx)
        for m, arr in preds.items():
            oof_preds[m][test_idx] = arr

    # Article-level Spearman for paired Wilcoxon test across the 100 independent articles
    art_sp = {m: np.array([safe_corr(spearmanr, oof_preds[m][a], H_target[a]) for a in range(100)]) for m in method_names}

    # Article-cluster bootstrap draws over N=100 independent articles
    boot = {m: {"item_sp": [], "sys_sp": [], "item_r": [], "item_tau": [], "icc_a1": [], "mae": [], "bias": []} for m in method_names}
    for _ in range(n_boot):
        b_idx = rng.integers(0, 100, size=100)
        h_b = H_target[b_idx]
        h_b_flat = h_b.ravel()
        h_b_sys = h_b.mean(axis=0)
        for m in method_names:
            p_b = oof_preds[m][b_idx]
            p_b_flat = p_b.ravel()
            boot[m]["item_sp"].append(safe_corr(spearmanr, p_b_flat, h_b_flat))
            boot[m]["item_tau"].append(safe_corr(kendalltau, p_b_flat, h_b_flat))
            boot[m]["item_r"].append(safe_corr(pearsonr, p_b_flat, h_b_flat))
            boot[m]["sys_sp"].append(safe_corr(spearmanr, p_b.mean(axis=0), h_b_sys))
            ia, _ = icc_a1_and_c1(p_b_flat, h_b_flat)
            boot[m]["icc_a1"].append(ia)
            boot[m]["mae"].append(float(np.mean(np.abs(p_b_flat - h_b_flat))))
            boot[m]["bias"].append(float(np.mean(p_b_flat - h_b_flat)))

    summary = {}
    h_flat = H_target.ravel()
    h_sys = H_target.mean(axis=0)
    for m in method_names:
        p = oof_preds[m]
        p_flat = p.ravel()
        ia, ic = icc_a1_and_c1(p_flat, h_flat)
        w_p_item = float(wilcoxon(art_sp[m], art_sp["Uniform Mean"]).pvalue) if m != "Uniform Mean" else 1.0
        # Also test significance vs Naive SVD on article-level Spearman
        w_p_svd = float(wilcoxon(art_sp[m], art_sp["Naive SVD"]).pvalue) if m != "Naive SVD" else 1.0
        summary[m] = {
            "item_spearman": safe_corr(spearmanr, p_flat, h_flat),
            "item_spearman_ci": [float(np.quantile(boot[m]["item_sp"], 0.025)), float(np.quantile(boot[m]["item_sp"], 0.975))],
            "item_kendall": safe_corr(kendalltau, p_flat, h_flat),
            "item_kendall_ci": [float(np.quantile(boot[m]["item_tau"], 0.025)), float(np.quantile(boot[m]["item_tau"], 0.975))],
            "item_pearson": safe_corr(pearsonr, p_flat, h_flat),
            "item_pearson_ci": [float(np.quantile(boot[m]["item_r"], 0.025)), float(np.quantile(boot[m]["item_r"], 0.975))],
            "sys_spearman": safe_corr(spearmanr, p.mean(axis=0), h_sys),
            "sys_spearman_ci": [float(np.quantile(boot[m]["sys_sp"], 0.025)), float(np.quantile(boot[m]["sys_sp"], 0.975))],
            "sys_kendall": safe_corr(kendalltau, p.mean(axis=0), h_sys),
            "sys_pearson": safe_corr(pearsonr, p.mean(axis=0), h_sys),
            "icc_a1": ia,
            "icc_a1_ci": [float(np.quantile(boot[m]["icc_a1"], 0.025)), float(np.quantile(boot[m]["icc_a1"], 0.975))],
            "icc_c1": ic,
            "mae": float(np.mean(np.abs(p_flat - h_flat))),
            "mae_ci": [float(np.quantile(boot[m]["mae"], 0.025)), float(np.quantile(boot[m]["mae"], 0.975))],
            "mean_bias": float(np.mean(p_flat - h_flat)),
            "mean_bias_ci": [float(np.quantile(boot[m]["bias"], 0.025)), float(np.quantile(boot[m]["bias"], 0.975))],
            "mean_pred": float(p_flat.mean()),
            "mean_human": float(h_flat.mean()),
            "within_art_spearman_mean": float(art_sp[m].mean()),
            "wilcoxon_within_art_vs_mean_p": w_p_item,
            "wilcoxon_within_art_vs_naive_svd_p": w_p_svd,
        }
    return summary


def module4_summeval_regimes(bundle: dict, rng: np.random.Generator) -> dict:
    """Run held-out 5-fold SummEval evaluation across Clean, Clone-Hijacking, Adversarial, and Wild regimes."""
    G4 = np.stack(
        [
            bundle["G_dims"]["con"][:, :, 0],
            bundle["G_dims"]["rel"][:, :, 0],
            bundle["G_dims"]["coh"][:, :, 0],
            bundle["G_dims"]["flu"][:, :, 0],
        ],
        axis=2,
    )  # (100, 16, 4)
    B_fam = bundle["B_rf"]  # (100, 16, 4)
    B_en = bundle["B_en"]  # (100, 16, 70)
    H = bundle["H_overall"]

    # Map BART/word_len linearly onto raw range only at pool assembly? No, _fit_and_predict_aggregators handles raw scales!
    W_len = bundle["word_len"][:, :, None]

    # Regime A: Clean heterogeneous multi-family (4 GPT4 + 4 Seq2Seq + 4 BART-en + 1 length)
    pool_clean = np.concatenate([G4, B_fam, B_en[:, :, :4], W_len], axis=2)

    # Regime B: Clone-Hijacking Sweep (4 GPT4 + M BART prompt clones, M in 0..64)
    clone_sweep = []
    for M in (0, 2, 4, 8, 16, 32, 64):
        pool_m = G4 if M == 0 else np.concatenate([G4, B_en[:, :, :M]], axis=2)
        res_m = eval_5fold_summeval(pool_m, H, [0, 1, 2, 3], rng, n_boot=250)
        clone_sweep.append({"M_clones": M, "K_total": 4 + M, "results": res_m})

    # Regime C1: Asymmetric Clone-Flood Adversaries (4 GPT4 + 4 Seq2Seq + 16 inverted clones -> 66.7% raw adv!)
    adv_clone_base = -B_fam[:, :, 2]  # inverted BART-CNN
    adv_16 = np.stack([adv_clone_base + rng.normal(0, 0.02, size=(100, 16)) for _ in range(16)], axis=2)
    pool_cloned_adv = np.concatenate([G4, B_fam, adv_16], axis=2)

    # Regime C2: Diverse Independent Adversaries Sweep on Clean Pool (delta in 0.10..0.44)
    adv_sweep = []
    honest_base = np.concatenate([G4, B_fam, B_en[:, :, :2]], axis=2)  # K_h = 10
    for k_adv in (1, 3, 5, 8):
        # Mix of inverted seq2seq & inverted GPT4 calls
        pool_a_list = [-honest_base[:, :, j % honest_base.shape[2]] for j in range(k_adv)]
        pool_a = np.concatenate([honest_base, np.stack(pool_a_list, axis=2)], axis=2)
        res_a = eval_5fold_summeval(pool_a, H, [0, 1, 2, 3], rng, n_boot=250)
        adv_sweep.append({"k_adv": k_adv, "delta_raw": k_adv / (10 + k_adv), "results": res_a})

    # Regime D: Wild Messy Kitchen-Sink Pool (K=38: 4 GPT4 + 4 Seq2Seq + 20 BART clones + 3 inv + 6 noise + 1 len)
    noise_6 = rng.uniform(0, 1, size=(100, 16, 6))
    inv_3 = np.stack([-G4[:, :, 0], -B_fam[:, :, 0], -B_fam[:, :, 2]], axis=2)
    pool_wild = np.concatenate([G4, B_fam, B_en[:, :, :20], inv_3, noise_6, W_len], axis=2)

    # Also 200 repeated 50/50 split robustness check on Wild Kitchen-Sink & 64-Clone regimes
    split_robust = {}
    for label, P in (("Cloned_M64", np.concatenate([G4, B_en[:, :, :64]], axis=2)), ("Wild_K38", pool_wild)):
        diffs_sys = []
        diffs_item = []
        rb_sys_list = []
        mean_sys_list = []
        naive_sys_list = []
        for _ in range(N_SPLITS_ROBUST):
            tr_idx = rng.choice(100, size=50, replace=False)
            mask = np.ones(100, dtype=bool)
            mask[tr_idx] = False
            te_idx = np.where(mask)[0]
            preds = _fit_and_predict_aggregators(P[tr_idx], P[te_idx], [0, 1, 2, 3])
            h_te = H[te_idx]
            rb_s = safe_corr(spearmanr, preds["RB-Spec TVD-MI"].mean(0), h_te.mean(0))
            mn_s = safe_corr(spearmanr, preds["Uniform Mean"].mean(0), h_te.mean(0))
            nv_s = safe_corr(spearmanr, preds["Naive SVD"].mean(0), h_te.mean(0))
            rb_i = safe_corr(spearmanr, preds["RB-Spec TVD-MI"].ravel(), h_te.ravel())
            mn_i = safe_corr(spearmanr, preds["Uniform Mean"].ravel(), h_te.ravel())
            rb_sys_list.append(rb_s)
            mean_sys_list.append(mn_s)
            naive_sys_list.append(nv_s)
            diffs_sys.append(rb_s - mn_s)
            diffs_item.append(rb_i - mn_i)
        split_robust[label] = {
            "rb_sys_sp_mean": float(np.mean(rb_sys_list)),
            "rb_sys_sp_ci": [float(np.quantile(rb_sys_list, 0.025)), float(np.quantile(rb_sys_list, 0.975))],
            "mean_sys_sp_mean": float(np.mean(mean_sys_list)),
            "mean_sys_sp_ci": [float(np.quantile(mean_sys_list, 0.025)), float(np.quantile(mean_sys_list, 0.975))],
            "naive_svd_sys_sp_mean": float(np.mean(naive_sys_list)),
            "naive_svd_sys_sp_ci": [float(np.quantile(naive_sys_list, 0.025)), float(np.quantile(naive_sys_list, 0.975))],
            "wilcoxon_sys_rb_vs_mean_p": float(wilcoxon(diffs_sys).pvalue),
            "wilcoxon_item_rb_vs_mean_p": float(wilcoxon(diffs_item).pvalue),
        }

    return {
        "regimeA_clean_k13": eval_5fold_summeval(pool_clean, H, [0, 1, 2, 3], rng),
        "regimeB_clone_sweep": clone_sweep,
        "regimeC1_cloned_adv_67pct_raw": eval_5fold_summeval(pool_cloned_adv, H, [0, 1, 2, 3], rng),
        "regimeC2_adv_sweep": adv_sweep,
        "regimeD_wild_k38": eval_5fold_summeval(pool_wild, H, [0, 1, 2, 3], rng),
        "splits_200_50_50": split_robust,
    }


def _extract_mtbench_features(df: pd.DataFrame) -> np.ndarray:
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
    vec.fit(prompts + resp_a + resp_b)
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

    rng_mt = np.random.default_rng(SEED + 99)
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
        jitter = rng_mt.normal(0.0, 0.05 * base.std(), size=len(base))
        cols.append(base + jitter)
    cols.extend([-gpt4_score, -tfidf_diff, -lex_diff])
    return np.column_stack(cols)


def module5_mt_bench_pairwise(rng: np.random.Generator) -> dict:
    """Evaluate pairwise aggregators on LMSYS MT-Bench human judgments across 80 questions."""
    df_h = pd.read_parquet(MTBENCH_CACHE / "human.parquet")
    df_g = pd.read_parquet(MTBENCH_CACHE / "gpt4.parquet")
    k_cols = ["question_id", "model_a", "model_b", "turn"]
    df_g_flip = df_g.copy()
    df_g_flip["model_a"], df_g_flip["model_b"] = df_g["model_b"], df_g["model_a"]
    df_g_flip["winner"] = df_g["winner"].replace({"model_a": "model_b", "model_b": "model_a"})
    df_g_all = pd.concat([df_g, df_g_flip], ignore_index=True).groupby(k_cols, as_index=False).first()

    merged = df_h.merge(df_g_all[["question_id", "model_a", "model_b", "turn", "winner"]], on=k_cols, suffixes=("_h", "_g"))
    # Keep decisive human judgments (model_a vs model_b) for accuracy & balanced model win-rate evaluation
    dec = merged[merged["winner_h"].isin(["model_a", "model_b"])].reset_index(drop=True)
    y_true = np.where(dec["winner_h"] == "model_a", 1.0, -1.0)
    q_ids = dec["question_id"].to_numpy()
    unique_qs = np.array(sorted(dec["question_id"].unique()))
    assert len(unique_qs) == 80 and len(dec) == 2575

    # Inter-human ceiling across repeated non-tie ratings
    agree_pairs = []
    for _, grp in dec.groupby(k_cols):
        if len(grp) >= 2:
            vals = grp["winner_h"].tolist()
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    agree_pairs.append(int(vals[i] == vals[j]))
    inter_human_acc = float(np.mean(agree_pairs))

    X_raw = _extract_mtbench_features(dec)  # (2575, 17)
    K = X_raw.shape[1]

    # 5-fold question-disjoint CV so every question is scored out-of-sample
    q_perm = rng.permutation(unique_qs)
    q_folds = np.array_split(q_perm, N_FOLDS)
    methods = ["GPT4 Only", "Length Clone Baseline", "Majority Vote", "Uniform Mean", "Naive Pairwise SVD", "RB-Spec Pairwise"]
    oof_scores = {m: np.zeros(len(dec), dtype=np.float64) for m in methods}

    for q_te in q_folds:
        te_mask = np.isin(q_ids, q_te)
        tr_mask = ~te_mask
        X_tr, X_te = X_raw[tr_mask], X_raw[te_mask]
        sd = np.maximum(X_tr.std(axis=0), 1e-8)
        # Note: pairwise differences naturally center around 0 under symmetry; scale by train RMS
        rms = np.maximum(np.sqrt(np.mean(X_tr ** 2, axis=0)), 1e-8)
        Z_tr = X_tr / rms
        Z_te = X_te / rms

        # Baseline 1 & 2: single GPT-4 and verbosity length baseline
        oof_scores["GPT4 Only"][te_mask] = Z_te[:, 0]
        oof_scores["Length Clone Baseline"][te_mask] = Z_te[:, 3]
        oof_scores["Majority Vote"][te_mask] = np.sign(Z_te).mean(axis=1)
        oof_scores["Uniform Mean"][te_mask] = Z_te.mean(axis=1)

        # Naive SVD on Z_tr
        _, _, vt_nv = np.linalg.svd(Z_tr, full_matrices=False)
        v_nv = vt_nv[0] * (1.0 if np.sum(vt_nv[0]) >= 0 else -1.0)
        oof_scores["Naive Pairwise SVD"][te_mask] = Z_te @ v_nv

        # RB-Spec Pairwise TVD-MI: correlation balancing over K judges on Z_tr (unsupervised!)
        R = np.nan_to_num(np.corrcoef(Z_tr, rowvar=False), nan=0.0)
        eff_mass = np.maximum(np.sum(np.abs(R) ** 6, axis=1), 1.0)
        bal = eff_mass ** (-0.5)
        Z_bal = Z_tr * bal[None, :]
        _, _, vt_rb = np.linalg.svd(Z_bal, full_matrices=False)
        v_rb = vt_rb[0] * bal
        if np.sum(v_rb * bal) < 0:
            v_rb = -v_rb
        oof_scores["RB-Spec Pairwise"][te_mask] = Z_te @ v_rb

    # Model-level win rates across the 6 chat models
    models = sorted(dec["model_a"].unique())
    m_to_idx = {m: i for i, m in enumerate(models)}
    idx_a = np.array([m_to_idx[x] for x in dec["model_a"]])
    idx_b = np.array([m_to_idx[x] for x in dec["model_b"]])

    def win_rates(votes_sign: np.ndarray) -> np.ndarray:
        wins = np.zeros(len(models))
        counts = np.zeros(len(models))
        for k_i in range(len(models)):
            mask_a = idx_a == k_i
            mask_b = idx_b == k_i
            wins[k_i] += np.sum(votes_sign[mask_a] > 0) + np.sum(votes_sign[mask_b] < 0)
            counts[k_i] += np.sum(mask_a) + np.sum(mask_b)
        return wins / np.maximum(counts, 1.0)

    human_wr = win_rates(y_true)
    # Cluster bootstrap over 80 questions
    q_to_rows = [np.where(q_ids == q)[0] for q in unique_qs]
    q_acc = {
        m: np.array([float(np.mean((oof_scores[m][rows] * y_true[rows]) > 0)) for rows in q_to_rows])
        for m in methods
    }

    summary = {"n_pairs": int(len(dec)), "n_questions": 80, "inter_human_agreement": inter_human_acc, "K_panel": K, "methods": {}}
    for m in methods:
        pred_sign = np.where(oof_scores[m] > 0, 1.0, np.where(oof_scores[m] < 0, -1.0, 0.0))
        acc_overall = float(np.mean(pred_sign == y_true))
        wr = win_rates(pred_sign)
        boot_acc = []
        for _ in range(N_BOOT):
            b_qs = rng.integers(0, 80, size=80)
            rows_b = np.concatenate([q_to_rows[qi] for qi in b_qs])
            boot_acc.append(float(np.mean(pred_sign[rows_b] == y_true[rows_b])))
        w_p_mean = float(wilcoxon(q_acc[m], q_acc["Uniform Mean"]).pvalue) if m != "Uniform Mean" else 1.0
        w_p_svd = float(wilcoxon(q_acc[m], q_acc["Naive Pairwise SVD"]).pvalue) if m != "Naive Pairwise SVD" else 1.0
        summary["methods"][m] = {
            "pairwise_acc": acc_overall,
            "pairwise_acc_ci": [float(np.quantile(boot_acc, 0.025)), float(np.quantile(boot_acc, 0.975))],
            "model_wr_spearman": safe_corr(spearmanr, wr, human_wr),
            "model_wr_kendall": safe_corr(kendalltau, wr, human_wr),
            "model_wr_pearson": safe_corr(pearsonr, wr, human_wr),
            "wilcoxon_q_vs_mean_p": w_p_mean,
            "wilcoxon_q_vs_naive_svd_p": w_p_svd,
        }
    return summary


def module6_exact_theory_and_cost_frontier(bundle: dict, rng: np.random.Generator) -> dict:
    """Verify Lean 4 NewTheory scaling identities numerically (500 MC reps) + Cost-Aware Nitzan-Paroush."""
    # 1. Clone Hijacking vs Balanced 1/F Recovery (500 reps)
    # F=6 diverse families; tau1_sq=0.25 (bad family flooded with M clones), other 5 families tau_f^2=0.01, sigma_f^2=0.05
    clone_rows = []
    for M in (1, 2, 4, 8, 16, 32, 64, 128):
        # Simulate N_MC realizations of hierarchical random effects
        # Family 0: K_0 = M clones around family bias ~ N(0, 0.5^2), idio ~ N(0, sqrt(0.05)^2)
        # Families 1..5: 1 judge each, family bias ~ N(0, 0.1^2), idio ~ N(0, sqrt(0.05)^2)
        fam0_bias = rng.normal(0.0, 0.5, size=(N_MC, 1))
        fam0_scores = fam0_bias + rng.normal(0.0, np.sqrt(0.05), size=(N_MC, M))
        other_scores = rng.normal(0.0, np.sqrt(0.01 + 0.05), size=(N_MC, 5))
        all_scores = np.hstack([fam0_scores, other_scores])
        unif_mean = all_scores.mean(axis=1)
        # Balanced family weights: each of F=6 families gets 1/6 total weight -> fam0 members get 1/(6*M)
        w_bal = np.array([1.0 / (6.0 * M)] * M + [1.0 / 6.0] * 5)
        bal_mean = all_scores @ w_bal
        th_unif = float((M ** 2 * 0.25 + M * 0.05 + 5 * (0.01 + 0.05)) / ((M + 5) ** 2))
        th_bal = float((0.25 + 0.05 / M + 5 * (0.01 + 0.05)) / 36.0)
        clone_rows.append({
            "M": M,
            "unif_var_mc": float(np.var(unif_mean)),
            "unif_var_theory": th_unif,
            "bal_var_mc": float(np.var(bal_mean)),
            "bal_var_theory": th_bal,
        })

    # 2. Effective-Family Adversarial Threshold: 12 cloned adversaries in 1 family (raw 12/16=75% > 50%) vs 4 honest families (delta_fam=1/5=20%!)
    K_honest_fam = 4
    gamma = 0.25
    adv_clone_counts = (1, 2, 4, 8, 12, 24)
    fam_adv_rows = []
    for k_adv in adv_clone_counts:
        # Each family's latent competence correlation sign: 4 honest (+), 1 adversarial family with k_adv clones (-)
        # Simulate N_MC binary questions, honest fam correctness p=0.75 (correlated 0.7 within fam), adv fam p=0.25 (cloned rho=0.95)
        true_bit = rng.integers(0, 2, size=N_MC)
        # 4 honest judges (1 per fam)
        h_flip = rng.uniform(0, 1, size=(N_MC, K_honest_fam)) < (0.5 + gamma)
        h_votes = np.where(h_flip, true_bit[:, None], 1 - true_bit[:, None]) * 2.0 - 1.0
        # 1 adversarial family cloned k_adv times
        a_base = np.where(rng.uniform(0, 1, size=(N_MC, 1)) < (0.5 - gamma), true_bit[:, None], 1 - true_bit[:, None]) * 2.0 - 1.0
        a_votes = np.repeat(a_base, k_adv, axis=1)
        votes = np.hstack([h_votes, a_votes])
        maj_acc = float(np.mean(((votes.mean(axis=1) > 0) == (true_bit == 1))))
        w_fam = np.array([1.0 / 5.0] * K_honest_fam + [1.0 / (5.0 * k_adv)] * k_adv)
        bal_acc = float(np.mean((((votes @ w_fam) > 0) == (true_bit == 1))))
        fam_adv_rows.append({
            "k_adv_clones": k_adv,
            "delta_raw": float(k_adv / (K_honest_fam + k_adv)),
            "delta_fam": 0.20,
            "raw_majority_acc": maj_acc,
            "balanced_family_acc": bal_acc,
        })

    # 3. Real SummEval Cost-Aware Nitzan-Paroush Pareto Frontier (c_GPT4 = 20, c_Seq2Seq = 1)
    # Split 50/50 article CV evaluating Budget B in {20, 40, 60, 80, 100} USD units
    G4 = np.stack([bundle["G_dims"][d][:, :, 0] for d in ("con", "rel", "coh", "flu")], axis=2)  # cost 20 each
    B4 = bundle["B_rf"]  # cost 1 each
    H = bundle["H_overall"]
    tr_idx = np.arange(0, 50)
    te_idx = np.arange(50, 100)
    cost_rows = []
    for n_gpt in (0, 1, 2, 3, 4):
        for n_seq in (0, 2, 4):
            if n_gpt == 0 and n_seq == 0:
                continue
            budget = 20 * n_gpt + 1 * n_seq
            cols = [G4[:, :, :n_gpt], B4[:, :, :n_seq]]
            P = np.concatenate([c for c in cols if c.shape[2] > 0], axis=2)
            # Fit RB-Spec (unsupervised) AND Nitzan-Paroush anchor-calibrated weights (10 train anchor articles)
            preds = _fit_and_predict_aggregators(P[tr_idx], P[te_idx], list(range(n_gpt)))
            rb_sp = safe_corr(spearmanr, preds["RB-Spec TVD-MI"].mean(0), H[te_idx].mean(0))
            unif_sp = safe_corr(spearmanr, preds["Uniform Mean"].mean(0), H[te_idx].mean(0))
            # Nitzan-Paroush inverse-variance competence weighting using 10 train anchor articles
            Z_tr_c, Z_te_c, _, _ = _standardize_fold(P[:10], P[te_idx])
            h_tr = ((H[:10] - H[:10].mean()) / (H[:10].std() + 1e-8)).ravel()
            z_tr_f = Z_tr_c.reshape(-1, P.shape[2])
            mu_j = np.maximum((z_tr_f.T @ h_tr) / len(h_tr), 0.01)
            resid_var = np.maximum(np.var(z_tr_f - mu_j[None, :] * h_tr[:, None], axis=0), 0.05)
            w_np = mu_j / resid_var
            s_np = (Z_te_c.reshape(-1, P.shape[2]) @ w_np).reshape(len(te_idx), 16)
            np_sp = safe_corr(spearmanr, s_np.mean(0), H[te_idx].mean(0))
            cost_rows.append({
                "n_gpt4": n_gpt,
                "n_seq2seq": n_seq,
                "cost_units": budget,
                "rb_spec_sys_spearman": rb_sp,
                "uniform_sys_spearman": unif_sp,
                "nitzan_paroush_sys_spearman": np_sp,
            })

    return {
        "clone_hijacking_mc": clone_rows,
        "effective_family_adversarial_mc": fam_adv_rows,
        "cost_frontier_summeval": cost_rows,
    }


def make_6panel_figure(mod2: dict, mod3: dict, mod4: dict, mod5: dict, mod6: dict) -> None:
    """Create 6-panel publication figure with visible explicit subplot gridlines."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    plt.subplots_adjust(hspace=0.36, wspace=0.28)

    for ax in axes.ravel():
        ax.grid(True, which="both", linestyle=":", linewidth=0.65, alpha=0.75)

    # Panel A: SummEval Copy Blindness & Shift Invariance
    ax = axes[0, 0]
    labels_a = ["Verbatim Copy\n(Prompt TVD)", "Top Real Sys\n(Prompt TVD)", "Verbatim Copy\n(Peer TVD)", "Top Real Sys\n(Peer TVD)"]
    vals_a = [
        mod2["verbatim_copy_prompt_tvd"],
        mod2["max_real_prompt_tvd"],
        mod2["verbatim_copy_peer_tvd"],
        mod2["top_real_peer_tvd"],
    ]
    colors_a = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(range(4), vals_a, color=colors_a, alpha=0.88, width=0.62)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_a, fontsize=8)
    ax.set_ylabel("TF-IDF TVD-MI Witness Score")
    ax.set_title("A. Real SummEval: Copy Blindness vs Peer")
    for b, v in zip(bars, vals_a):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel B: SummEval Clone-Hijacking Sweep (M=0..64 BART clones + 4 GPT-4)
    ax = axes[0, 1]
    ms = [r["M_clones"] for r in mod4["regimeB_clone_sweep"]]
    rb_sys = [r["results"]["RB-Spec TVD-MI"]["sys_spearman"] for r in mod4["regimeB_clone_sweep"]]
    nv_sys = [r["results"]["Naive SVD"]["sys_spearman"] for r in mod4["regimeB_clone_sweep"]]
    mn_sys = [r["results"]["Uniform Mean"]["sys_spearman"] for r in mod4["regimeB_clone_sweep"]]
    md_sys = [r["results"]["Itemwise Median"]["sys_spearman"] for r in mod4["regimeB_clone_sweep"]]
    ax.plot(ms, rb_sys, "o-", color="#2ca02c", lw=2.2, label="RB-Spec TVD-MI (Ours)")
    ax.plot(ms, mn_sys, "s--", color="#1f77b4", lw=1.7, label="Uniform Mean")
    ax.plot(ms, nv_sys, "^--", color="#d62728", lw=1.7, label="Naive SVD")
    ax.plot(ms, md_sys, "d:", color="#9467bd", lw=1.6, label="Itemwise Median")
    ax.set_xlabel("Redundant BART-CNN Clones Added (M)")
    ax.set_ylabel("Held-Out System Spearman (16 systems)")
    ax.set_title("B. SummEval Clone-Hijacking (Inverse Scaling)")
    ax.set_ylim(0.72, 1.01)
    ax.legend(loc="lower left", framealpha=0.92)

    # Panel C: SummEval Null Overfitting (Naive vs Split-Sample, n=20 articles)
    ax = axes[0, 2]
    n20 = [r for r in mod3["null_overfitting_rows"] if r["n_art"] == 20]
    ks = [r["K"] for r in n20]
    nv_m = [r["naive_mean"] for r in n20]
    nv_lo = [r["naive_ci"][0] for r in n20]
    nv_hi = [r["naive_ci"][1] for r in n20]
    sp_m = [r["split_mean"] for r in n20]
    sp_lo = [r["split_ci"][0] for r in n20]
    sp_hi = [r["split_ci"][1] for r in n20]
    ax.plot(ks, nv_m, "o-", color="#d62728", lw=2.0, label="Naive In-Sample Max")
    ax.fill_between(ks, nv_lo, nv_hi, color="#d62728", alpha=0.15)
    ax.plot(ks, sp_m, "s-", color="#2ca02c", lw=2.0, label="Split-Sample Cross-Fitted")
    ax.fill_between(ks, sp_lo, sp_hi, color="#2ca02c", alpha=0.15)
    ax.axhline(0.0, color="black", linestyle="--", lw=1.0)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Critic Pool Size K (log2 scale, n=20 articles)")
    ax.set_ylabel("Null Correlation Witness")
    ax.set_title("C. SummEval Null Overfitting Elimination")
    ax.legend(loc="upper left", framealpha=0.92)

    # Panel D: SummEval Asymmetric Cloned-Adversary & Wild Kitchen-Sink Pools
    ax = axes[1, 0]
    m_list = ["Single GPT4-con", "Uniform Mean", "Itemwise Median", "Naive SVD", "RB-Spec TVD-MI"]
    short_lbl = ["GPT4-1", "Mean", "Median", "NaiveSVD", "RB-Spec"]
    c1 = [mod4["regimeC1_cloned_adv_67pct_raw"][m]["item_spearman"] for m in m_list]
    d_w = [mod4["regimeD_wild_k38"][m]["item_spearman"] for m in m_list]
    x = np.arange(len(m_list))
    ax.bar(x - 0.18, d_w, width=0.35, color="#1f77b4", alpha=0.88, label="Wild K=38 (clones+adv+noise)")
    ax.bar(x + 0.18, c1, width=0.35, color="#ff7f0e", alpha=0.88, label="Adv-Flood 66.7% Raw (1 fam)")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(short_lbl, fontsize=8.5)
    ax.set_ylabel("Held-Out Item Spearman (1,600 items)")
    ax.set_title("D. SummEval Held-Out Wild & Cloned-Adv Pools")
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.92)

    # Panel E: LMSYS MT-Bench Real Pairwise Human Preference (80 questions, 2,575 pairs)
    ax = axes[1, 1]
    mt_order = ["Length Clone Baseline", "Majority Vote", "Uniform Mean", "Naive Pairwise SVD", "GPT4 Only", "RB-Spec Pairwise"]
    mt_lbl = ["LenClone", "MajVote", "UnifMean", "NaiveSVD", "GPT4-1", "RB-Spec"]
    accs = [mod5["methods"][m]["pairwise_acc"] for m in mt_order]
    lo_e = [accs[i] - mod5["methods"][m]["pairwise_acc_ci"][0] for i, m in enumerate(mt_order)]
    hi_e = [mod5["methods"][m]["pairwise_acc_ci"][1] - accs[i] for i, m in enumerate(mt_order)]
    cols_e = ["#7f7f7f", "#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]
    ax.bar(range(len(mt_order)), accs, yerr=[lo_e, hi_e], capsize=3.5, color=cols_e, alpha=0.88, width=0.6)
    ax.axhline(mod5["inter_human_agreement"], color="black", linestyle="--", lw=1.2, label=f"Inter-Human ({mod5['inter_human_agreement']:.3f})")
    ax.set_xticks(range(len(mt_order)))
    ax.set_xticklabels(mt_lbl, fontsize=8)
    ax.set_ylim(0.50, 0.90)
    ax.set_ylabel("Held-Out Pairwise Accuracy")
    ax.set_title("E. LMSYS MT-Bench Confounded Pool (K=17)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    # Panel F: Exact Hierarchical 1/F Variance vs Clone-Hijacking (MC vs Lean Theory)
    ax = axes[1, 2]
    ms_mc = [r["M"] for r in mod6["clone_hijacking_mc"]]
    ax.plot(ms_mc, [r["unif_var_mc"] for r in mod6["clone_hijacking_mc"]], "o", color="#d62728", label="Uniform MC")
    ax.plot(ms_mc, [r["unif_var_theory"] for r in mod6["clone_hijacking_mc"]], "-", color="#d62728", lw=1.8, label="Uniform Theory (tends to tau1^2)")
    ax.plot(ms_mc, [r["bal_var_mc"] for r in mod6["clone_hijacking_mc"]], "s", color="#2ca02c", label="Balanced 1/F MC")
    ax.plot(ms_mc, [r["bal_var_theory"] for r in mod6["clone_hijacking_mc"]], "-", color="#2ca02c", lw=1.8, label="Balanced Theory (<= B/F)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Clone Multiplicity M in Weak Family (log2)")
    ax.set_ylabel("Ensemble Variance")
    ax.set_title("F. Lean Theory vs 500-Rep MC (F=6 Families)")
    ax.legend(loc="center right", fontsize=7.8, framealpha=0.92)

    fig.savefig(RESULTS_DIR / "exp13_real_llm_and_theory_6panel.png")
    fig.savefig(RESULTS_DIR / "exp13_real_llm_and_theory_6panel.pdf")
    plt.close(fig)


def write_summary_csvs(summary: dict) -> None:
    """Persist machine-readable CSV tables for key regimes."""
    with (RESULTS_DIR / "summeval_heldout_regimes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["regime", "method", "item_sp", "item_sp_lo", "item_sp_hi", "sys_sp", "sys_sp_lo", "sys_sp_hi", "icc_a1", "mae", "mean_bias", "wilcoxon_p_vs_mean"])
        for reg_key in ("regimeA_clean_k13", "regimeC1_cloned_adv_67pct_raw", "regimeD_wild_k38"):
            for m, d in summary["module4_summeval"][reg_key].items():
                w.writerow([
                    reg_key, m,
                    round(d["item_spearman"], 4), round(d["item_spearman_ci"][0], 4), round(d["item_spearman_ci"][1], 4),
                    round(d["sys_spearman"], 4), round(d["sys_spearman_ci"][0], 4), round(d["sys_spearman_ci"][1], 4),
                    round(d["icc_a1"], 4), round(d["mae"], 4), round(d["mean_bias"], 4),
                    f"{d['wilcoxon_within_art_vs_mean_p']:.3e}",
                ])

    with (RESULTS_DIR / "summeval_clone_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["M_clones", "K_total", "method", "item_sp", "sys_sp", "icc_a1", "mean_bias"])
        for row in summary["module4_summeval"]["regimeB_clone_sweep"]:
            for m, d in row["results"].items():
                w.writerow([row["M_clones"], row["K_total"], m, round(d["item_spearman"], 4), round(d["sys_spearman"], 4), round(d["icc_a1"], 4), round(d["mean_bias"], 4)])

    with (RESULTS_DIR / "mtbench_pairwise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "pairwise_acc", "acc_lo", "acc_hi", "wr_spearman", "wr_kendall", "wr_pearson", "wilcoxon_vs_mean_p", "wilcoxon_vs_naive_svd_p"])
        for m, d in summary["module5_mtbench"]["methods"].items():
            w.writerow([
                m, round(d["pairwise_acc"], 4), round(d["pairwise_acc_ci"][0], 4), round(d["pairwise_acc_ci"][1], 4),
                round(d["model_wr_spearman"], 4), round(d["model_wr_kendall"], 4), round(d["model_wr_pearson"], 4),
                f"{d['wilcoxon_q_vs_mean_p']:.3e}", f"{d['wilcoxon_q_vs_naive_svd_p']:.3e}",
            ])


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    print("Loading SummEval SHA-256 verified archive...")
    bundle = load_summeval_bundle()

    print("Module 1: Computing human ceiling...")
    ceiling = compute_human_ceiling(bundle, rng)

    print("Module 2: Prompt-copy falsification & double-centering shift invariance...")
    mod2 = module2_falsification_and_double_centering(bundle)

    print("Module 3: Within-family error correlation vs cross-family & null overfitting...")
    mod3 = module3_saturation_and_split_null(bundle, rng)

    print("Module 4: 5-fold article-disjoint + 200x 50/50 SummEval evaluation...")
    mod4 = module4_summeval_regimes(bundle, rng)

    print("Module 5: LMSYS MT-Bench 5-fold question-disjoint pairwise evaluation...")
    mod5 = module5_mt_bench_pairwise(rng)

    print("Module 6: Lean theory MC verifications & cost-aware frontier...")
    mod6 = module6_exact_theory_and_cost_frontier(bundle, rng)

    summary = {
        "seed": SEED,
        "elapsed_sec": round(time.time() - t0, 2),
        "summeval_parse_counts": bundle["parse_counts"],
        "module1_human_ceiling": ceiling,
        "module2_falsification": mod2,
        "module3_saturation_and_null": mod3,
        "module4_summeval": mod4,
        "module5_mtbench": mod5,
        "module6_theory_and_cost": mod6,
    }
    (RESULTS_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csvs(summary)
    make_6panel_figure(mod2, mod3, mod4, mod5, mod6)
    print(f"ALL MODULES COMPLETE in {summary['elapsed_sec']}s")


if __name__ == "__main__":
    main()
