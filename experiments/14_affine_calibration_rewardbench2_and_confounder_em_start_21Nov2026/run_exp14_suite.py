"""Experiment 14 — Semi-Supervised Affine & Platt Calibration, RewardBench 2 Multi-Family
Evaluation, Cross-Family Confounder Recovery (CR-RB-Spec & EM), and Davis-Kahan Bounds.

Usage:
    python3 run_exp14_suite.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import kendalltau, pearsonr, spearmanr, wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

matplotlib.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "text.usetex": False,
    }
)

EXP14_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP14_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMEVAL_CACHE = Path("/tmp/corr-ce-public-ordinal/summeval")
MTBENCH_CACHE = Path("/tmp/mt_bench_public")
RB2_CACHE = Path("/tmp/rewardbench2_public")

SEED = 42
N_FOLDS = 5
N_BOOT = 2000
N_SPLITS_CAL = 200
N_MC = 500

GRADE_LINE = re.compile(r"\A\s*([1-5])(?:[ \t]*/[ \t]*5)?[ \t]*(?:\r?\n|\Z)")

RB2_MAIN_MODELS = [
    # Skywork family (8)
    "Skywork/Skywork-Reward-Llama-3.1-8B",
    "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "Skywork/Skywork-Reward-V2-Llama-3.2-1B",
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B",
    "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    "Skywork/Skywork-Reward-V2-Qwen3-8B",
    # AllenAI family (5)
    "allenai/Llama-3.1-8B-Base-RM-RB2",
    "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    "allenai/Llama-3.1-Tulu-3-8B-RM",
    "allenai/Llama-3.1-Tulu-3-8B-SFT-RM-RB2",
    "allenai/Llama-3.1-Tulu-3-8B-DPO-RM-RB2",
    # GRM family (4)
    "Ray2333/GRM-Llama3-8B-rewardmodel-ft",
    "Ray2333/GRM-gemma2-2B-rewardmodel-ft",
    "Ray2333/GRM-llama3-8B-distill",
    "Ray2333/GRM-llama3-8B-sftreg",
    # OpenAssistant legacy weak family (3)
    "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
    "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    # Diverse strong singletons (3)
    "Qwen/WorldPM-72B",
    "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1",
    "NCSOFT/Llama-3-OffsetBias-RM-8B",
]


class NumericUnpickler(pickle.Unpickler):
    """Restricted unpickler matching verified BARTScore SummEval archive."""

    def find_class(self, module: str, name: str):
        from numpy.core.multiarray import scalar

        allowed = {
            ("numpy.core.multiarray", "scalar"): scalar,
            ("numpy", "dtype"): np.dtype,
        }
        return allowed[(module, name)]


def icc_a1_and_c1(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Shrout-Fleiss two-way random single-measure ICC(A,1) and ICC(C,1)."""
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
    """Safe rank/linear correlation for near-constant vectors."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    val = float(fn(a, b).statistic)
    return 0.0 if np.isnan(val) else val


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """15-bin Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y[mask].mean()) - float(probs[mask].mean()))
    return float(ece)


def fit_rb_spec_weights(C_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit redundancy-balanced spectral weight vector w_rb strictly from unlabeled centered C_tr."""
    K = C_tr.shape[1]
    sd = np.maximum(C_tr.std(axis=0), 1e-8)
    C_std = C_tr / sd
    R = np.nan_to_num(np.corrcoef(C_std, rowvar=False), nan=0.0)
    if np.ndim(R) < 2:
        R = np.ones((1, 1))
    eff_mass = np.maximum(np.sum(np.abs(R) ** 6, axis=1), 1.0)
    bal = eff_mass ** (-0.5)
    C_bal = C_std * bal[None, :]
    _, _, vt_rb = np.linalg.svd(C_bal, full_matrices=False)
    v_rb = vt_rb[0] * bal
    if np.sum(v_rb * bal) < 0:
        v_rb = -v_rb
    # Combine standardization into effective weight over raw centered scores
    eff_w = v_rb / sd
    return eff_w, v_rb


def fit_residualized_rb_spec(
    C_tr: np.ndarray,
    C_te: np.ndarray,
    cov_tr: np.ndarray,
    cov_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CR-RB-Spec: orthogonalize C against observable confounders cov fit strictly on train fold."""
    # Add intercept just in case covariates aren't strictly zero-mean
    X_tr = np.column_stack([np.ones(len(cov_tr)), cov_tr])
    X_te = np.column_stack([np.ones(len(cov_te)), cov_te])
    beta, _, _, _ = np.linalg.lstsq(X_tr, C_tr, rcond=None)
    R_tr = C_tr - X_tr @ beta
    R_te = C_te - X_te @ beta
    w_cr, _ = fit_rb_spec_weights(R_tr)
    return R_tr @ w_cr, R_te @ w_cr, w_cr


def load_summeval_bundle() -> dict:
    """Load SummEval annotations, G-Eval GPT-4 scores, and BART/PRISM metrics from local cache."""
    annotations = [
        json.loads(line)
        for line in (SUMMEVAL_CACHE / "model_annotations.aligned.jsonl").read_bytes().splitlines()
    ]
    raw = {(r["id"], r["model_id"]): r for r in annotations}
    docs = sorted({a for a, _ in raw})
    systems = sorted({s for _, s in raw})
    assert len(docs) == 100 and len(systems) == 16

    with (SUMMEVAL_CACHE / "bart_metrics.pkl").open("rb") as f:
        bart = NumericUnpickler(f).load()

    dims = ["consistency", "relevance", "coherence", "fluency"]
    experts = np.zeros((100, 16, 3, 4), dtype=np.float64)
    for di, doc in enumerate(docs):
        for si, sys_id in enumerate(systems):
            for ei, ann in enumerate(raw[(doc, sys_id)]["expert_annotations"]):
                for ki, dim in enumerate(dims):
                    experts[di, si, ei, ki] = float(ann[dim])
    H_overall = experts.mean(axis=(2, 3))

    G_dims = {}
    for short, fn in (
        ("con", "gpt4_con_detailed.json"),
        ("rel", "gpt4_rel_detailed.json"),
        ("coh", "gpt4_coh_detailed.json"),
        ("flu", "gpt4_flu_detailed.json"),
    ):
        recs = {(r["doc_id"], r["system_id"]): r for r in json.loads((SUMMEVAL_CACHE / fn).read_bytes())}
        M = np.full((100, 16, 20), np.nan, dtype=np.float64)
        for di, doc in enumerate(docs):
            for si, sys_id in enumerate(systems):
                for j, txt in enumerate(recs[(doc, sys_id)]["all_responses"][:20]):
                    m = GRADE_LINE.match(str(txt))
                    if m:
                        M[di, si, j] = float(m.group(1))
                row = M[di, si]
                if np.isnan(row).any():
                    row[np.isnan(row)] = np.nanmean(row) if not np.isnan(row).all() else 3.0
        G_dims[short] = M

    s0 = bart[docs[0]]["sys_summs"][systems[0]]["scores"]
    en_keys = sorted([k for k in s0 if k.startswith("bart_score_cnn_src_hypo_en_")])
    rf_keys = ["bart_score_src_hypo", "bart_score_para_src_hypo", "bart_score_cnn_src_hypo", "prism_src_hypo"]
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

    # Prompt-candidate lexical TF-IDF cosine similarity for confounder residualization covariate
    all_texts = X_docs + [Y_cand[di][si] for di in range(100) for si in range(16)]
    vec = TfidfVectorizer(max_features=12000, stop_words="english").fit(all_texts)
    X_mat = vec.transform(X_docs)
    Y_mat = vec.transform([Y_cand[di][si] for di in range(100) for si in range(16)])
    prompt_sim = np.zeros((100, 16), dtype=np.float64)
    for di in range(100):
        rows = slice(di * 16, (di + 1) * 16)
        prompt_sim[di] = np.asarray(Y_mat[rows].multiply(X_mat[di]).sum(axis=1)).ravel()

    G4 = np.stack([G_dims[d][:, :, 0] for d in ("con", "rel", "coh", "flu")], axis=2)
    J_wild = np.concatenate([G4, B_rf, B_en[:, :, :29], word_len[:, :, None]], axis=2)
    return {
        "H_overall": H_overall,
        "G4": G4,
        "B_rf": B_rf,
        "B_en": B_en,
        "word_len": word_len,
        "prompt_sim": prompt_sim,
        "J_wild": J_wild,
    }


def load_mtbench_bundle() -> dict:
    """Load LMSYS MT-Bench K=17 confounded pairwise matrix + length covariate."""
    df_h = pd.read_parquet(MTBENCH_CACHE / "human.parquet")
    df_g = pd.read_parquet(MTBENCH_CACHE / "gpt4.parquet")
    k_cols = ["question_id", "model_a", "model_b", "turn"]
    df_g_flip = df_g.copy()
    df_g_flip["model_a"], df_g_flip["model_b"] = df_g["model_b"], df_g["model_a"]
    df_g_flip["winner"] = df_g["winner"].replace({"model_a": "model_b", "model_b": "model_a"})
    df_g_all = pd.concat([df_g, df_g_flip], ignore_index=True).groupby(k_cols, as_index=False).first()

    merged = df_h.merge(df_g_all[k_cols + ["winner"]], on=k_cols, suffixes=("_h", "_g"))
    dec = merged[merged["winner_h"].isin(["model_a", "model_b"])].reset_index(drop=True)
    y_true = np.where(dec["winner_h"] == "model_a", 1.0, -1.0)
    assert len(dec) == 2575 and dec["question_id"].nunique() == 80

    prompts, resp_a, resp_b = [], [], []
    for _, row in dec.iterrows():
        turn_idx = 0 if int(row["turn"]) == 1 else 1
        ca, cb = row["conversation_a"], row["conversation_b"]
        prompts.append(str(ca[2 * turn_idx]["content"]))
        resp_a.append(str(ca[2 * turn_idx + 1]["content"]))
        resp_b.append(str(cb[2 * turn_idx + 1]["content"]))

    vec = TfidfVectorizer(max_features=3000, stop_words="english").fit(prompts + resp_a + resp_b)
    VP = vec.transform(prompts)
    VA = vec.transform(resp_a)
    VB = vec.transform(resp_b)
    sim_diff = np.asarray(VP.multiply(VA).sum(axis=1) - VP.multiply(VB).sum(axis=1)).ravel()

    def lex_div(txt: str) -> float:
        toks = re.findall(r"[a-z]+", txt.lower())
        return len(set(toks)) / np.sqrt(len(toks) + 1.0)

    div_diff = np.array([lex_div(a) - lex_div(b) for a, b in zip(resp_a, resp_b)], dtype=np.float64)
    wlen_diff = np.array([len(a.split()) - len(b.split()) for a, b in zip(resp_a, resp_b)], dtype=np.float64)
    clen_diff = np.array([len(a) - len(b) for a, b in zip(resp_a, resp_b)], dtype=np.float64)

    gmap = {"model_a": 1.0, "model_b": -1.0, "tie": 0.0, "tie (inconsistent)": 0.0}
    gpt4_score = np.array([gmap[w] for w in dec["winner_g"]], dtype=np.float64)

    rng_mt = np.random.default_rng(SEED + 99)
    cols = [gpt4_score, sim_diff, div_diff]
    for c_idx in range(11):
        base = wlen_diff if c_idx % 2 == 0 else clen_diff
        jitter = rng_mt.normal(0.0, 0.05 * base.std(), size=len(base))
        cols.append(base + jitter)
    cols.extend([-gpt4_score, -sim_diff, -div_diff])
    X_raw = np.column_stack(cols)
    return {
        "X_raw": X_raw,
        "y_true": y_true,
        "q_ids": dec["question_id"].to_numpy(dtype=int),
        "len_diff": wlen_diff,
        "sim_diff": sim_diff,
    }


def load_rewardbench2_bundle() -> dict:
    """Load deduplicated RewardBench 2 matrix (1,753 prompts x candidate completions) from /tmp cache."""
    meta = json.load((RB2_CACHE / "aligned_meta.json").open("r", encoding="utf-8"))
    models_data = {}
    for m in RB2_MAIN_MODELS + ["PKU-Alignment/beaver-7b-v1.0-cost", "PKU-Alignment/beaver-7b-v1.0-reward", "PKU-Alignment/beaver-7b-v2.0-cost", "PKU-Alignment/beaver-7b-v2.0-reward"]:
        slug = m.replace("/", "__")
        models_data[m] = json.load((RB2_CACHE / "compact" / f"{slug}.json").open("r", encoding="utf-8"))

    n_prompts = len(meta)
    subsets = np.array([r["subset"] for r in meta])
    num_correct = np.array([r["num_correct"] for r in meta], dtype=int)
    tot_comp = np.array([r["total_completions"] for r in meta], dtype=int)
    unique_tots = sorted(set(tot_comp.tolist()))

    # Flatten candidates across all prompts with explicit prompt index tracking
    cand_prompt_idx = []
    cand_is_chosen = []
    cand_words = []
    prompt_cand_slices = []
    cursor = 0
    for pi, r in enumerate(meta):
        k_c = r["total_completions"]
        nc = r["num_correct"]
        prompt_cand_slices.append((cursor, cursor + k_c, nc))
        cand_prompt_idx.extend([pi] * k_c)
        cand_is_chosen.extend([1] * nc + [0] * (k_c - nc))
        cw = r["cand_words"]
        if len(cw) != k_c:
            cw = (cw + [0] * k_c)[:k_c]
        cand_words.extend(cw)
        cursor += k_c

    n_cands = cursor
    all_model_names = list(models_data.keys())
    S_raw = np.zeros((n_cands, len(all_model_names)), dtype=np.float64)
    S_centered = np.zeros((n_cands, len(all_model_names)), dtype=np.float64)
    for mi, m in enumerate(all_model_names):
        sc_rows = models_data[m]["scores"]
        for pi, (s, e, _) in enumerate(prompt_cand_slices):
            vals = np.asarray(sc_rows[pi], dtype=np.float64)
            S_raw[s:e, mi] = vals
            # Within-prompt Z-normalized scores prevent huge logit-scale RMs from dominating
            sd_p = max(vals.std(), 1e-6)
            S_centered[s:e, mi] = (vals - vals.mean()) / sd_p

    # Save byte-for-byte item trace for RewardBench 2
    r0 = meta[0]
    trace = {
        "prompt_subset": r0["subset"],
        "prompt_id": r0["id"],
        "num_correct": r0["num_correct"],
        "total_completions": r0["total_completions"],
        "cand_word_counts": r0["cand_words"],
        "skywork_v2_8b_scores": models_data["Skywork/Skywork-Reward-V2-Llama-3.1-8B"]["scores"][0],
        "pku_beaver_v1_reward_scores": models_data["PKU-Alignment/beaver-7b-v1.0-reward"]["scores"][0],
        "unique_completion_counts": unique_tots,
        "n_prompts_dedup": n_prompts,
        "n_candidates_total": n_cands,
        "raw_sha256_meta": hashlib.sha256((RB2_CACHE / "aligned_meta.json").read_bytes()).hexdigest(),
    }
    (RESULTS_DIR / "trace_one_item_rb2.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")

    return {
        "meta": meta,
        "all_models": all_model_names,
        "S_raw": S_raw,
        "S_centered": S_centered,
        "subsets": subsets,
        "num_correct": num_correct,
        "tot_comp": tot_comp,
        "prompt_cand_slices": prompt_cand_slices,
        "cand_prompt_idx": np.asarray(cand_prompt_idx, dtype=int),
        "cand_is_chosen": np.asarray(cand_is_chosen, dtype=int),
        "cand_words": np.asarray(cand_words, dtype=np.float64),
    }


def module1_summeval_calibration(summeval: dict, rng: np.random.Generator) -> dict:
    """Module 1: 5-to-20 anchor affine calibration on SummEval Wild K=38 closing ICC(A,1) -> ICC(C,1)."""
    J_wild = summeval["J_wild"]  # (100, 16, 38)
    H = summeval["H_overall"]    # (100, 16)
    anchor_m_list = [0, 2, 5, 10, 20, 50]

    records = {m: {k: [] for k in ("item_sp", "sys_sp", "bias", "mae", "rmse", "icc_a1", "icc_c1")} for m in anchor_m_list}
    mean_baseline_records = {m: {k: [] for k in ("icc_a1", "mae", "bias")} for m in anchor_m_list}

    for _ in range(N_SPLITS_CAL):
        perm = rng.permutation(100)
        tr_idx, te_idx = perm[:50], perm[50:]
        J_tr, J_te = J_wild[tr_idx], J_wild[te_idx]
        H_tr, H_te = H[tr_idx], H[te_idx]

        # Fit unsupervised RB-Spec strictly on unlabeled 50 train articles
        C_tr = (J_tr - J_tr.mean(axis=1, keepdims=True)).reshape(-1, J_wild.shape[2])
        C_te = (J_te - J_te.mean(axis=1, keepdims=True)).reshape(-1, J_wild.shape[2])
        w_rb, _ = fit_rb_spec_weights(C_tr)
        s_tr = (C_tr @ w_rb).reshape(50, 16)
        s_te = (C_te @ w_rb).reshape(50, 16)
        z_mu, z_sd = float(s_tr.mean()), max(float(s_tr.std()), 1e-8)
        z_tr = (s_tr - z_mu) / z_sd
        z_te = (s_te - z_mu) / z_sd

        # Also Uniform Mean (train min-max rescaled to [1,5])
        tr_f = J_tr.reshape(-1, J_wild.shape[2])
        mn, mx = tr_f.min(0), np.maximum(tr_f.max(0), tr_f.min(0) + 1e-8)
        u_tr = (1.0 + 4.0 * np.clip((tr_f - mn) / (mx - mn), 0.0, 1.0)).mean(1).reshape(50, 16)
        u_te = (1.0 + 4.0 * np.clip((J_te.reshape(-1, J_wild.shape[2]) - mn) / (mx - mn), 0.0, 1.0)).mean(1).reshape(50, 16)

        # Uncalibrated G-Eval raw [1,5] anchor prior (zero human labels)
        raw_gpt = J_tr[:, :, :4]
        mu_gpt = float(raw_gpt.mean())
        sd_gpt = max(float(raw_gpt.std()), 1e-3)

        h_flat = H_te.ravel()
        h_sys = H_te.mean(axis=0)
        anc_order = rng.permutation(50)

        for m in anchor_m_list:
            if m == 0:
                pred_rb = np.clip(mu_gpt + sd_gpt * z_te, 1.0, 5.0)
                pred_unif = u_te
            else:
                a_idx = anc_order[:m]
                z_a = z_tr[a_idx].ravel()
                h_a = H_tr[a_idx].ravel()
                # Fit monotone OLS affine transform (a_hat * z + b_hat), clamping slope >= 0.05
                cov_zh = float(np.mean((z_a - z_a.mean()) * (h_a - h_a.mean())))
                var_z = max(float(np.var(z_a)), 1e-6)
                a_hat = max(cov_zh / var_z, 0.05)
                b_hat = float(h_a.mean() - a_hat * z_a.mean())
                pred_rb = np.clip(a_hat * z_te + b_hat, 1.0, 5.0)

                ua = u_tr[a_idx].ravel()
                cov_uh = float(np.mean((ua - ua.mean()) * (h_a - h_a.mean())))
                au = max(cov_uh / max(float(np.var(ua)), 1e-6), 0.05)
                bu = float(h_a.mean() - au * ua.mean())
                pred_unif = np.clip(au * u_te + bu, 1.0, 5.0)

            pf = pred_rb.ravel()
            ia, ic = icc_a1_and_c1(pf, h_flat)
            records[m]["item_sp"].append(safe_corr(spearmanr, pf, h_flat))
            records[m]["sys_sp"].append(safe_corr(spearmanr, pred_rb.mean(0), h_sys))
            records[m]["bias"].append(float(np.mean(pf - h_flat)))
            records[m]["mae"].append(float(np.mean(np.abs(pf - h_flat))))
            records[m]["rmse"].append(float(np.sqrt(np.mean((pf - h_flat) ** 2))))
            records[m]["icc_a1"].append(ia)
            records[m]["icc_c1"].append(ic)

            uf = pred_unif.ravel()
            uia, _ = icc_a1_and_c1(uf, h_flat)
            mean_baseline_records[m]["icc_a1"].append(uia)
            mean_baseline_records[m]["mae"].append(float(np.mean(np.abs(uf - h_flat))))
            mean_baseline_records[m]["bias"].append(float(np.mean(uf - h_flat)))

    # 5-fold article-disjoint CV with m=10 anchors for paired Wilcoxon across 100 independent articles
    folds = np.array_split(rng.permutation(100), N_FOLDS)
    oof_m0 = np.zeros((100, 16))
    oof_m10 = np.zeros((100, 16))
    for te_idx in folds:
        mask = np.ones(100, dtype=bool)
        mask[te_idx] = False
        tr_idx = np.where(mask)[0]
        J_tr, J_te = J_wild[tr_idx], J_wild[te_idx]
        H_tr = H[tr_idx]
        C_tr = (J_tr - J_tr.mean(axis=1, keepdims=True)).reshape(-1, J_wild.shape[2])
        C_te = (J_te - J_te.mean(axis=1, keepdims=True)).reshape(-1, J_wild.shape[2])
        w_rb, _ = fit_rb_spec_weights(C_tr)
        s_tr = (C_tr @ w_rb).reshape(len(tr_idx), 16)
        s_te = (C_te @ w_rb).reshape(len(te_idx), 16)
        z_tr = (s_tr - s_tr.mean()) / max(s_tr.std(), 1e-8)
        z_te = (s_te - s_tr.mean()) / max(s_tr.std(), 1e-8)
        raw_gpt = J_tr[:, :, :4]
        oof_m0[te_idx] = np.clip(float(raw_gpt.mean()) + max(float(raw_gpt.std()), 1e-3) * z_te, 1.0, 5.0)
        anc = np.arange(10)
        z_a, h_a = z_tr[anc].ravel(), H_tr[anc].ravel()
        a_hat = max(float(np.mean((z_a - z_a.mean()) * (h_a - h_a.mean()))) / max(float(np.var(z_a)), 1e-6), 0.05)
        b_hat = float(h_a.mean() - a_hat * z_a.mean())
        oof_m10[te_idx] = np.clip(a_hat * z_te + b_hat, 1.0, 5.0)

    art_mae_0 = np.mean(np.abs(oof_m0 - H), axis=1)
    art_mae_10 = np.mean(np.abs(oof_m10 - H), axis=1)
    art_mse_0 = np.mean((oof_m0 - H) ** 2, axis=1)
    art_mse_10 = np.mean((oof_m10 - H) ** 2, axis=1)
    p_mae_0_vs_10 = float(wilcoxon(art_mae_10, art_mae_0).pvalue)
    p_mse_0_vs_10 = float(wilcoxon(art_mse_10, art_mse_0).pvalue)

    rows = []
    for m in anchor_m_list:
        row = {"m_anchors": m}
        for k, vals in records[m].items():
            row[k] = float(np.mean(vals))
            row[f"{k}_lo"] = float(np.quantile(vals, 0.025))
            row[f"{k}_hi"] = float(np.quantile(vals, 0.975))
        row["unif_icc_a1"] = float(np.mean(mean_baseline_records[m]["icc_a1"]))
        row["unif_mae"] = float(np.mean(mean_baseline_records[m]["mae"]))
        rows.append(row)

    return {
        "splits_200": rows,
        "cv_m0_vs_m10_wilcoxon": {
            "oof_m0_icc_a1": icc_a1_and_c1(oof_m0.ravel(), H.ravel())[0],
            "oof_m10_icc_a1": icc_a1_and_c1(oof_m10.ravel(), H.ravel())[0],
            "oof_m0_mae": float(art_mae_0.mean()),
            "oof_m10_mae": float(art_mae_10.mean()),
            "wilcoxon_mae_p": p_mae_0_vs_10,
            "wilcoxon_mse_p": p_mse_0_vs_10,
        },
    }


def _eval_rb2_pool(
    S_c: np.ndarray,
    S_raw_panel: np.ndarray,
    rb2: dict,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict:
    """Evaluate 5-fold prompt-disjoint CV across multi-judge aggregators on RewardBench 2."""
    n_prompts = len(rb2["meta"])
    slices = rb2["prompt_cand_slices"]
    c_pidx = rb2["cand_prompt_idx"]
    subsets = rb2["subsets"]
    uniq_subsets = sorted(set(subsets.tolist()))

    folds = np.array_split(rng.permutation(n_prompts), N_FOLDS)
    methods = ["Best Single", "Uniform Z-Mean", "Itemwise Median", "Naive SVD", "RB-Spec TVD-MI"]
    n_cands = len(c_pidx)
    oof_cand = {m: np.zeros(n_cands, dtype=np.float64) for m in methods}

    for te_p in folds:
        te_mask_p = np.zeros(n_prompts, dtype=bool)
        te_mask_p[te_p] = True
        tr_cand_mask = ~te_mask_p[c_pidx]
        te_cand_mask = te_mask_p[c_pidx]

        C_tr = S_c[tr_cand_mask]
        C_te = S_c[te_cand_mask]

        # Best single RM on RB2 (index 2: Skywork-V2-Llama-3.1-8B) for ceiling reference
        oof_cand["Best Single"][te_cand_mask] = C_te[:, 2]
        oof_cand["Uniform Z-Mean"][te_cand_mask] = C_te.mean(axis=1)
        oof_cand["Itemwise Median"][te_cand_mask] = np.median(C_te, axis=1)

        _, _, vt_nv = np.linalg.svd(C_tr, full_matrices=False)
        v_nv = vt_nv[0] * (1.0 if np.sum(vt_nv[0]) >= 0 else -1.0)
        oof_cand["Naive SVD"][te_cand_mask] = C_te @ v_nv

        w_rb, _ = fit_rb_spec_weights(C_tr)
        oof_cand["RB-Spec TVD-MI"][te_cand_mask] = C_te @ w_rb

    # Prompt-level binary accuracy: min(chosen) > max(rejected) (matches official RB2 metric)
    p_acc = {m: np.zeros(n_prompts, dtype=np.float64) for m in methods}
    for m in methods:
        sc = oof_cand[m]
        for pi, (s, e, nc) in enumerate(slices):
            vals = sc[s:e]
            p_acc[m][pi] = 1.0 if float(vals[:nc].min()) > float(vals[nc:].max()) else 0.0

    sub_masks = {sub: (subsets == sub) for sub in uniq_subsets}
    out = {}
    for m in methods:
        arr = p_acc[m]
        boot_vals = []
        for _ in range(n_boot):
            bi = rng.integers(0, n_prompts, size=n_prompts)
            boot_vals.append(float(arr[bi].mean()))
        sub_acc = {sub: float(arr[sub_masks[sub]].mean()) for sub in uniq_subsets}
        macro_acc = float(np.mean(list(sub_acc.values())))
        w_vs_mean = float(wilcoxon(arr, p_acc["Uniform Z-Mean"]).pvalue) if m != "Uniform Z-Mean" else 1.0
        w_vs_svd = float(wilcoxon(arr, p_acc["Naive SVD"]).pvalue) if m != "Naive SVD" else 1.0
        out[m] = {
            "prompt_acc": float(arr.mean()),
            "prompt_acc_ci": [float(np.quantile(boot_vals, 0.025)), float(np.quantile(boot_vals, 0.975))],
            "macro_acc": macro_acc,
            "subset_acc": sub_acc,
            "wilcoxon_vs_mean_p": w_vs_mean,
            "wilcoxon_vs_naive_svd_p": w_vs_svd,
        }
    return {"summary": out, "oof_rb_cand": oof_cand["RB-Spec TVD-MI"]}


def module2_rewardbench2(rb2: dict, rng: np.random.Generator) -> dict:
    """Module 2: Multi-family RewardBench 2 regimes + m-anchor Platt probability calibration."""
    all_m = rb2["all_models"]
    idx_main = [all_m.index(m) for m in RB2_MAIN_MODELS]
    S_main_c = rb2["S_centered"][:, idx_main]
    S_main_r = rb2["S_raw"][:, idx_main]

    # Regime 1: Balanced-ish Natural Diverse 23-Model Pool
    res_natural = _eval_rb2_pool(S_main_c, S_main_r, rb2, rng, n_boot=N_BOOT)

    # Regime 2: Weak-Clone Flood (32 noisy clones of weak OpenAssistant DeBERTa/Pythia legacy judges)
    idx_oasst = [RB2_MAIN_MODELS.index(m) for m in (
        "OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1",
        "OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
        "OpenAssistant/reward-model-deberta-v3-large-v2",
    )]
    sweep_rows = []
    for M in (0, 8, 16, 32):
        if M == 0:
            S_p = S_main_c
        else:
            clones = [
                S_main_c[:, idx_oasst[j % 3]] + rng.normal(0.0, 0.12, size=len(S_main_c))
                for j in range(M)
            ]
            S_p = np.column_stack([S_main_c] + clones)
        res_m = _eval_rb2_pool(S_p, S_p, rb2, rng, n_boot=300)
        sweep_rows.append({"M_weak_clones": M, "K_total": S_p.shape[1], "metrics": res_m["summary"]})

    # Regime 3: Natural Cost/Negative-Polarity + Cloned Adversary Flood (>55% bad raw headcount)
    # Use naturally anti-correlated PKU-beaver-v1.0-reward (acc=0.1654, corr=-0.234) + cloned noisy variants
    idx_beaver_rew = all_m.index("PKU-Alignment/beaver-7b-v1.0-reward")
    b1_rew = rb2["S_centered"][:, idx_beaver_rew]
    # Add 28 noisy clones of natural inverted beaver-reward (out of 23 + 28 = 51 -> 54.9% inverted headcount)
    nat_adv_clones = [b1_rew + rng.normal(0.0, 0.12, size=len(S_main_c)) for _ in range(28)]
    S_nat_adv = np.column_stack([S_main_c] + nat_adv_clones)
    res_nat_adv = _eval_rb2_pool(S_nat_adv, S_nat_adv, rb2, rng, n_boot=N_BOOT)

    # Platt / logistic probability calibration over (chosen, rejected) pairs on raw logit mean & RB-Spec
    oof_rb = res_natural["oof_rb_cand"]
    raw_mean_cand = S_main_r.mean(axis=1)
    slices = rb2["prompt_cand_slices"]
    pair_prompt = []
    pair_margin_rb = []
    pair_margin_raw = []
    for pi, (s, e, nc) in enumerate(slices):
        sc_rb = oof_rb[s:e]
        sc_raw = raw_mean_cand[s:e]
        for ci in range(nc):
            for rj in range(nc, e - s):
                pair_prompt.append(pi)
                pair_margin_rb.append(float(sc_rb[ci] - sc_rb[rj]))
                pair_margin_raw.append(float(sc_raw[ci] - sc_raw[rj]))
    pair_prompt = np.asarray(pair_prompt, dtype=int)
    pair_margin_rb = np.asarray(pair_margin_rb, dtype=np.float64)
    pair_margin_raw = np.asarray(pair_margin_raw, dtype=np.float64)

    n_prompts = len(slices)
    anchor_counts = [0, 5, 10, 20, 50]
    cal_stats = {
        m: {"ece": [], "brier": [], "logloss": [], "acc": [], "raw_ece": [], "raw_brier": [], "raw_logloss": []}
        for m in anchor_counts
    }
    for _ in range(100):
        perm_p = rng.permutation(n_prompts)
        tr_p, te_p = perm_p[:200], perm_p[200:]
        te_mask = np.isin(pair_prompt, te_p)
        te_rb = pair_margin_rb[te_mask]
        te_raw = pair_margin_raw[te_mask]
        alt_sign = np.where((np.arange(len(te_rb)) % 2) == 0, 1.0, -1.0)
        X_te_rb = te_rb * alt_sign
        X_te_raw = te_raw * alt_sign
        y_te = (alt_sign > 0).astype(np.float64)

        for m in anchor_counts:
            if m == 0:
                probs_rb = expit(X_te_rb)
                probs_raw = expit(X_te_raw)
            else:
                anc_p = tr_p[:m]
                anc_mask = np.isin(pair_prompt, anc_p)
                am_rb = pair_margin_rb[anc_mask]
                am_raw = pair_margin_raw[anc_mask]
                y_fit = np.concatenate([np.ones(len(am_rb)), np.zeros(len(am_rb))])
                clf_rb = LogisticRegression(C=0.5, fit_intercept=False, max_iter=500).fit(
                    np.concatenate([am_rb, -am_rb])[:, None], y_fit
                )
                clf_raw = LogisticRegression(C=0.5, fit_intercept=False, max_iter=500).fit(
                    np.concatenate([am_raw, -am_raw])[:, None], y_fit
                )
                probs_rb = clf_rb.predict_proba(X_te_rb[:, None])[:, 1]
                probs_raw = clf_raw.predict_proba(X_te_raw[:, None])[:, 1]

            pr_c = np.clip(probs_rb, 1e-6, 1.0 - 1e-6)
            cal_stats[m]["ece"].append(expected_calibration_error(pr_c, y_te, n_bins=15))
            cal_stats[m]["brier"].append(float(np.mean((pr_c - y_te) ** 2)))
            cal_stats[m]["logloss"].append(float(-np.mean(y_te * np.log(pr_c) + (1.0 - y_te) * np.log(1.0 - pr_c))))
            cal_stats[m]["acc"].append(float(np.mean((pr_c >= 0.5) == (y_te == 1.0))))

            praw_c = np.clip(probs_raw, 1e-6, 1.0 - 1e-6)
            cal_stats[m]["raw_ece"].append(expected_calibration_error(praw_c, y_te, n_bins=15))
            cal_stats[m]["raw_brier"].append(float(np.mean((praw_c - y_te) ** 2)))
            cal_stats[m]["raw_logloss"].append(float(-np.mean(y_te * np.log(praw_c) + (1.0 - y_te) * np.log(1.0 - praw_c))))

    cal_summary = []
    for m in anchor_counts:
        row = {"m_anchor_prompts": m}
        for k, vals in cal_stats[m].items():
            row[k] = float(np.mean(vals))
            row[f"{k}_ci"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
        cal_summary.append(row)

    return {
        "n_prompts": n_prompts,
        "n_pairs": len(pair_margin_rb),
        "natural_23": res_natural["summary"],
        "weak_clone_sweep": sweep_rows,
        "natural_polarity_flood_51": res_nat_adv["summary"],
        "platt_calibration": cal_summary,
    }


def _dawid_skene_binary_em(votes_01: np.ndarray, w_init: np.ndarray | None = None, n_iter: int = 20) -> np.ndarray:
    """Cluster/Spectral-initialized Dawid-Skene EM over (N, K) binary votes in {0, 1}."""
    N, K = votes_01.shape
    if w_init is None:
        p = np.clip(votes_01.mean(axis=1), 0.05, 0.95)
    else:
        raw = votes_01 @ w_init
        raw_z = (raw - raw.mean()) / max(raw.std(), 1e-6)
        p = np.clip(expit(1.5 * raw_z), 0.05, 0.95)

    for _ in range(n_iter):
        pi_prior = float(np.clip(p.mean(), 0.05, 0.95))
        sum_p = max(float(p.sum()), 1e-6)
        sum_1p = max(float((1.0 - p).sum()), 1e-6)
        alpha = np.clip((p[:, None] * votes_01).sum(axis=0) / sum_p, 1e-3, 1.0 - 1e-3)
        beta = np.clip(((1.0 - p)[:, None] * (1.0 - votes_01)).sum(axis=0) / sum_1p, 1e-3, 1.0 - 1e-3)
        log_num = np.log(pi_prior) + votes_01 @ np.log(alpha) + (1.0 - votes_01) @ np.log(1.0 - alpha)
        log_den = np.log(1.0 - pi_prior) + votes_01 @ np.log(1.0 - beta) + (1.0 - votes_01) @ np.log(beta)
        p = np.clip(expit(log_num - log_den), 1e-6, 1.0 - 1e-6)
    return p


def module3_confounder_breakdown_and_recovery(
    summeval: dict, mtbench: dict, rb2: dict, rng: np.random.Generator
) -> dict:
    """Module 3: Universal cross-family confounder breakdown + CR-RB-Spec & EM recovery."""
    c_levels = [0.0, 0.5, 1.0, 1.5, 2.0]

    # 3A. SummEval Wild K=38 universal verbosity confounder
    J_wild = summeval["J_wild"].copy()
    H = summeval["H_overall"]
    wlen_se = summeval["word_len"]
    # Z-normalize word length & prompt overlap per article so universal bias creates strong within-prompt length correlation
    wlen_c = (wlen_se - wlen_se.mean(axis=1, keepdims=True))
    wlen_c = wlen_c / max(float(wlen_c.std()), 1e-8)
    psim_c = summeval["prompt_sim"] - summeval["prompt_sim"].mean(axis=1, keepdims=True)
    psim_c = psim_c / max(float(psim_c.std()), 1e-8)

    se_folds = np.array_split(rng.permutation(100), N_FOLDS)
    se_rows = []
    for c in c_levels:
        # Inject universal confounder proportional to judge within-prompt scale onto all 38 judges across all families!
        J_conf = J_wild.copy()
        c_base = J_wild - J_wild.mean(axis=1, keepdims=True)
        sd_j = np.maximum(c_base.reshape(-1, J_wild.shape[2]).std(axis=0), 1e-8)
        J_conf = J_conf + c * (wlen_c[:, :, None] * sd_j[None, None, :])

        preds = {k: np.zeros((100, 16)) for k in ("Uniform Mean", "Vanilla RB-Spec", "CR-RB-Spec", "RB-Spec + EM (5-anc)")}
        for te_idx in se_folds:
            mask = np.ones(100, dtype=bool); mask[te_idx] = False
            tr_idx = np.where(mask)[0]
            C_tr = (J_conf[tr_idx] - J_conf[tr_idx].mean(axis=1, keepdims=True)).reshape(-1, J_conf.shape[2])
            C_te = (J_conf[te_idx] - J_conf[te_idx].mean(axis=1, keepdims=True)).reshape(-1, J_conf.shape[2])

            preds["Uniform Mean"][te_idx] = (C_te / np.maximum(C_tr.std(0), 1e-8)).mean(1).reshape(len(te_idx), 16)
            w_rb, _ = fit_rb_spec_weights(C_tr)
            preds["Vanilla RB-Spec"][te_idx] = (C_te @ w_rb).reshape(len(te_idx), 16)

            cov_tr = np.column_stack([wlen_c[tr_idx].ravel(), psim_c[tr_idx].ravel()])
            cov_te = np.column_stack([wlen_c[te_idx].ravel(), psim_c[te_idx].ravel()])
            sc_cr_tr, sc_cr_te, _ = fit_residualized_rb_spec(C_tr, C_te, cov_tr, cov_te)
            preds["CR-RB-Spec"][te_idx] = sc_cr_te.reshape(len(te_idx), 16)

            # 2-factor top-2 spectral + 5 train anchor disambiguation
            sd_tr = np.maximum(C_tr.std(0), 1e-8)
            C_std = C_tr / sd_tr
            R = np.nan_to_num(np.corrcoef(C_std, rowvar=False), nan=0.0)
            bal = np.maximum(np.sum(np.abs(R) ** 6, axis=1), 1.0) ** (-0.5)
            _, _, vt2 = np.linalg.svd(C_std * bal[None, :], full_matrices=False)
            V2 = (vt2[:2].T * bal[:, None]) / sd_tr[:, None]  # (K, 2)
            F_tr = (C_tr @ V2).reshape(len(tr_idx), 16, 2)[:5].reshape(-1, 2)
            h_5_c = H[tr_idx[:5]] - H[tr_idx[:5]].mean(axis=1, keepdims=True)
            h_5 = h_5_c.ravel()
            beta2, _, _, _ = np.linalg.lstsq(F_tr, h_5, rcond=None)
            preds["RB-Spec + EM (5-anc)"][te_idx] = ((C_te @ V2) @ beta2).reshape(len(te_idx), 16)

        h_flat = H.ravel()
        res_c = {"c_confounder": c}
        for name, arr in preds.items():
            res_c[name] = safe_corr(spearmanr, arr.ravel(), h_flat)
        res_c["wilcoxon_cr_vs_vanilla_p"] = float(
            wilcoxon(
                [safe_corr(spearmanr, preds["CR-RB-Spec"][a], H[a]) for a in range(100)],
                [safe_corr(spearmanr, preds["Vanilla RB-Spec"][a], H[a]) for a in range(100)],
            ).pvalue
        )
        se_rows.append(res_c)

    # 3B. MT-Bench K=17 universal length confounder sweep
    X_mt = mtbench["X_raw"]
    y_mt = mtbench["y_true"]
    q_ids = mtbench["q_ids"]
    uniq_qs = sorted(set(q_ids.tolist()))
    mt_folds = np.array_split(rng.permutation(len(uniq_qs)), N_FOLDS)
    ld_mt = (mtbench["len_diff"] / max(float(mtbench["len_diff"].std()), 1e-8))[:, None]
    cov_mt = np.column_stack([mtbench["len_diff"], mtbench["sim_diff"]])

    mt_rows = []
    for c in c_levels:
        X_conf = X_mt + c * (ld_mt * np.maximum(X_mt.std(axis=0), 1e-8)[None, :])
        oof = {k: np.zeros(len(y_mt)) for k in ("Uniform Mean", "Vanilla RB-Spec", "CR-RB-Spec")}
        for f_idx in mt_folds:
            te_qs = set(uniq_qs[i] for i in f_idx)
            te_m = np.isin(q_ids, list(te_qs))
            tr_m = ~te_m
            rms = np.maximum(np.sqrt(np.mean(X_conf[tr_m] ** 2, axis=0)), 1e-8)
            Z_tr, Z_te = X_conf[tr_m] / rms, X_conf[te_m] / rms
            oof["Uniform Mean"][te_m] = Z_te.mean(axis=1)
            w_rb, _ = fit_rb_spec_weights(Z_tr)
            oof["Vanilla RB-Spec"][te_m] = Z_te @ w_rb
            _, sc_cr, _ = fit_residualized_rb_spec(Z_tr, Z_te, cov_mt[tr_m], cov_mt[te_m])
            oof["CR-RB-Spec"][te_m] = sc_cr

        q_rows = [np.where(q_ids == q)[0] for q in uniq_qs]
        res_m = {"c_confounder": c}
        for name, sc in oof.items():
            ps = np.where(sc > 0, 1.0, np.where(sc < 0, -1.0, 0.0))
            res_m[name] = float(np.mean(ps == y_mt))
        res_m["wilcoxon_cr_vs_vanilla_p"] = float(
            wilcoxon(
                [float(np.mean((oof["CR-RB-Spec"][r] * y_mt[r]) > 0)) for r in q_rows],
                [float(np.mean((oof["Vanilla RB-Spec"][r] * y_mt[r]) > 0)) for r in q_rows],
            ).pvalue
        )
        mt_rows.append(res_m)

    # 3C. 500-Repetition Monte Carlo Confounder + Adversarial Matrix (K=24, F=6, N=80 items)
    # 5 honest families (size 2, 2, 2, 4, 4) + 1 cloned adversary family (size 10) = 24 judges
    mc_rows = []
    fam_sizes = [2, 2, 2, 4, 4, 10]
    for c in (0.0, 0.5, 1.0, 1.5, 2.0):
        accs = {k: [] for k in ("Majority Vote", "Vanilla DS-EM", "Vanilla RB-Spec", "CR-RB-Spec", "CR-RB + DS-EM")}
        for _ in range(N_MC):
            n_it = 80
            z_true = rng.normal(0.0, 1.0, size=n_it)
            y_bin = (z_true > 0).astype(np.float64)
            w_conf = rng.normal(0.0, 1.0, size=n_it)
            w_obs = w_conf + rng.normal(0.0, 0.25, size=n_it)  # observable confounder proxy (r~0.97)
            cols = []
            for fi, kf in enumerate(fam_sizes):
                sign_f = -1.0 if fi == 5 else 1.0  # Family 5 is adversarial (10 clones > 41% headcount)
                u_f = rng.normal(0.0, 0.35, size=n_it)
                for _ in range(kf):
                    e_j = rng.normal(0.0, 0.45, size=n_it)
                    s_j = 0.9 * sign_f * z_true + c * w_conf + u_f + e_j
                    cols.append(s_j)
            M_cont = np.column_stack(cols)
            M_bin = (M_cont > 0).astype(np.float64)

            accs["Majority Vote"].append(float(np.mean((M_bin.mean(1) > 0.5) == (y_bin == 1.0))))
            p_ds = _dawid_skene_binary_em(M_bin, None)
            accs["Vanilla DS-EM"].append(float(np.mean((p_ds > 0.5) == (y_bin == 1.0))))

            w_van, _ = fit_rb_spec_weights(M_cont - M_cont.mean(0))
            accs["Vanilla RB-Spec"].append(float(np.mean((M_cont @ w_van > 0) == (y_bin == 1.0))))

            # CR-RB-Spec orthogonalizes against w_obs
            X_w = np.column_stack([np.ones(n_it), w_obs])
            b_w, _, _, _ = np.linalg.lstsq(X_w, M_cont, rcond=None)
            R_cont = M_cont - X_w @ b_w
            w_cr, _ = fit_rb_spec_weights(R_cont)
            sc_cr = R_cont @ w_cr
            accs["CR-RB-Spec"].append(float(np.mean((sc_cr > 0) == (y_bin == 1.0))))
            # Hybrid CR-RB-Spec initialization inside Dawid-Skene EM on residualized signs
            p_cr_ds = _dawid_skene_binary_em((R_cont > 0).astype(np.float64), w_cr)
            accs["CR-RB + DS-EM"].append(float(np.mean((p_cr_ds > 0.5) == (y_bin == 1.0))))

        mc_rows.append({"c_confounder": c, **{k: float(np.mean(v)) for k, v in accs.items()}})

    return {
        "summeval_confounder": se_rows,
        "mtbench_confounder": mt_rows,
        "mc_confounder_500": mc_rows,
    }


def module4_davis_kahan_rate(rb2: dict, rng: np.random.Generator) -> dict:
    """Module 4: Empirical Davis-Kahan eigenvector perturbation curve & O(1/n) sin^2(theta) slope."""
    all_m = rb2["all_models"]
    idx_main = [all_m.index(m) for m in RB2_MAIN_MODELS]
    C_full = rb2["S_centered"][:, idx_main]
    # Standardize population matrix across all 7,012 RewardBench 2 candidates for ground-truth spectral target
    sd_pop = np.maximum(C_full.std(axis=0), 1e-8)
    C_std_full = C_full / sd_pop
    R_pop = np.corrcoef(C_std_full, rowvar=False)
    bal_pop = np.maximum(np.sum(np.abs(R_pop) ** 6, axis=1), 1.0) ** (-0.5)
    Sigma_pop = (C_std_full * bal_pop[None, :]).T @ (C_std_full * bal_pop[None, :]) / len(C_std_full)
    eigvals, eigvecs = np.linalg.eigh(Sigma_pop)
    idx_sort = np.argsort(eigvals)[::-1]
    lam1, lam2 = float(eigvals[idx_sort[0]]), float(eigvals[idx_sort[1]])
    gamma_gap = lam1 - lam2
    u_pop = eigvecs[:, idx_sort[0]]

    n_grid = [10, 20, 40, 80, 160, 320, 640]
    n_prompts = len(rb2["prompt_cand_slices"])
    slices = rb2["prompt_cand_slices"]

    rb2_dk_rows = []
    for n_p in n_grid:
        sin_sqs = []
        bounds_lean = []
        eps_vals = []
        for _ in range(N_MC):
            # Sample i.i.d. with replacement from the empirical population (exact Sigma_pop) to avoid finite-population slope inflation
            sel_p = rng.choice(n_prompts, size=n_p, replace=True)
            c_rows = np.concatenate([np.arange(slices[p][0], slices[p][1]) for p in sel_p])
            C_sub = C_std_full[c_rows] * bal_pop[None, :]
            Sigma_hat = (C_sub.T @ C_sub) / len(c_rows)
            ev_h, vec_h = np.linalg.eigh(Sigma_hat)
            u_hat = vec_h[:, np.argmax(ev_h)]
            cos_th = min(abs(float(np.dot(u_hat, u_pop))), 1.0)
            sin_sq = max(0.0, 1.0 - cos_th ** 2)
            # Evaluate exact Rayleigh discrepancy epsilon in Lean Theorem `davis_kahan_rayleigh_sin_sq_bound`
            q_u1 = float(u_pop @ Sigma_pop @ u_pop)
            qh_u1 = float(u_pop @ Sigma_hat @ u_pop)
            q_uh = float(u_hat @ Sigma_pop @ u_hat)
            qh_uh = float(u_hat @ Sigma_hat @ u_hat)
            eps = max(abs(qh_u1 - q_u1), abs(qh_uh - q_uh))
            bound = 2.0 * eps / gamma_gap
            sin_sqs.append(sin_sq)
            eps_vals.append(eps)
            bounds_lean.append(bound)
        rb2_dk_rows.append({
            "n_prompts": n_p,
            "mean_sin_sq": float(np.mean(sin_sqs)),
            "sin_sq_ci": [float(np.quantile(sin_sqs, 0.025)), float(np.quantile(sin_sqs, 0.975))],
            "mean_sin": float(np.mean(np.sqrt(sin_sqs))),
            "mean_lean_bound": float(np.mean(bounds_lean)),
            "bound_violations": int(sum(s > b + 1e-12 for s, b in zip(sin_sqs, bounds_lean))),
        })

    # Fit log-log slope on sin^2(theta) vs n_prompts
    log_n = np.log(np.array([r["n_prompts"] for r in rb2_dk_rows], dtype=np.float64))
    log_s2 = np.log(np.array([r["mean_sin_sq"] for r in rb2_dk_rows], dtype=np.float64))
    log_b = np.log(np.array([r["mean_lean_bound"] for r in rb2_dk_rows], dtype=np.float64))
    slope_sin_sq = float(np.polyfit(log_n, log_s2, 1)[0])
    slope_bound = float(np.polyfit(log_n, log_b, 1)[0])

    return {
        "lambda1": lam1,
        "lambda2": lam2,
        "eigengap": gamma_gap,
        "rb2_dk_curve": rb2_dk_rows,
        "fitted_loglog_slope_sin_sq": slope_sin_sq,
        "fitted_loglog_slope_lean_bound": slope_bound,
    }


def make_figure(res_m1: dict, res_m2: dict, res_m3: dict, res_m4: dict) -> None:
    """Create crisp 6-panel Experiment 14 figure with major+minor gridlines."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    plt.subplots_adjust(hspace=0.36, wspace=0.28)

    # Panel A: SummEval m-Anchor Affine Calibration ICC(A,1) -> ICC(C,1) & MAE
    ax = axes[0, 0]
    rows_a = res_m1["splits_200"]
    ms = [r["m_anchors"] for r in rows_a]
    x_pos = np.arange(len(ms))
    icc_a = [r["icc_a1"] for r in rows_a]
    icc_c = [r["icc_c1"] for r in rows_a]
    mae_v = [r["mae"] for r in rows_a]
    ax.plot(x_pos, icc_c, "s--", color="#4c72b0", linewidth=1.8, label="Consistency ICC(C,1)")
    ax.plot(x_pos, icc_a, "o-", color="#2ca02c", linewidth=2.2, label="Agreement ICC(A,1) (RB-Spec)")
    ax.plot(x_pos, [r["unif_icc_a1"] for r in rows_a], "^:", color="#ff7f0e", linewidth=1.8, label="Agreement ICC(A,1) (Mean)")
    ax.plot(x_pos, mae_v, "d-.", color="#d62728", linewidth=1.8, label="MAE (RB-Spec)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(m) for m in ms])
    ax.set_xlabel("Human Anchor Articles m (of 50 Train)")
    ax.set_ylabel("Held-Out ICC / MAE")
    ax.set_title("A. SummEval Affine Calibration (200 Splits)")
    ax.set_ylim(0.25, 1.25)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.3)

    # Panel B: RewardBench 2 Multi-Family Regimes & Natural Polarity Flood
    ax = axes[0, 1]
    reg_names = ["Natural\n(K=23)", "Weak Clones\n(K=55)", "PKU Polarity\n(K=51)"]
    r_nat = res_m2["natural_23"]
    r_wk = res_m2["weak_clone_sweep"][-1]["metrics"]
    r_adv = res_m2["natural_polarity_flood_51"]
    m_keys = ["Uniform Z-Mean", "Itemwise Median", "Naive SVD", "RB-Spec TVD-MI"]
    colors = ["#4c72b0", "#ff7f0e", "#9467bd", "#2ca02c"]
    x_g = np.arange(len(reg_names))
    w_b = 0.19
    for idx_m, mk in enumerate(m_keys):
        vals = [r_nat[mk]["prompt_acc"], r_wk[mk]["prompt_acc"], r_adv[mk]["prompt_acc"]]
        ax.bar(x_g + (idx_m - 1.5) * w_b, vals, width=w_b, color=colors[idx_m], label=mk)
    ax.axhline(0.25, color="black", linestyle=":", linewidth=1.1, label="Random (0.25)")
    ax.set_xticks(x_g)
    ax.set_xticklabels(reg_names, fontsize=8.5)
    ax.set_ylabel("Held-Out Best-of-4 Accuracy")
    ax.set_title("B. RewardBench 2 (1,753 Non-Tie Prompts)")
    ax.set_ylim(0.0, 0.98)
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.0)

    # Panel C: RewardBench 2 m-Anchor Platt Probability Calibration (ECE & Brier)
    ax = axes[0, 2]
    cal_rows = res_m2["platt_calibration"]
    xc = np.arange(len(cal_rows))
    ax.plot(xc, [r["raw_ece"] for r in cal_rows], "^--", color="#ff7f0e", linewidth=2.0, label="Raw Logits 15-Bin ECE")
    ax.plot(xc, [r["ece"] for r in cal_rows], "o-", color="#d62728", linewidth=2.2, label="RB-Spec 15-Bin ECE")
    ax.plot(xc, [r["raw_brier"] for r in cal_rows], "v:", color="#9467bd", linewidth=1.8, label="Raw Logits Brier")
    ax.plot(xc, [r["brier"] for r in cal_rows], "s--", color="#1f77b4", linewidth=2.0, label="RB-Spec Brier")
    ax.set_xticks(xc)
    ax.set_xticklabels([str(r["m_anchor_prompts"]) for r in cal_rows])
    ax.set_xlabel("Platt Calibration Anchor Prompts m")
    ax.set_ylabel("Held-Out Probability Error")
    ax.set_title("C. RewardBench 2 Platt Calibration")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.2)

    # Panel D: SummEval Universal Verbosity Confounder & CR-RB-Spec Recovery
    ax = axes[1, 0]
    se_c = res_m3["summeval_confounder"]
    cs = [r["c_confounder"] for r in se_c]
    ax.plot(cs, [r["CR-RB-Spec"] for r in se_c], "o-", color="#2ca02c", linewidth=2.2, label="CR-RB-Spec (Ours)")
    ax.plot(cs, [r["RB-Spec + EM (5-anc)"] for r in se_c], "d-.", color="#17becf", linewidth=1.9, label="2-Factor RB + 5 Anchors")
    ax.plot(cs, [r["Vanilla RB-Spec"] for r in se_c], "s--", color="#d62728", linewidth=2.0, label="Vanilla RB-Spec (Breaks)")
    ax.plot(cs, [r["Uniform Mean"] for r in se_c], "^:", color="#4c72b0", linewidth=1.8, label="Uniform Mean")
    ax.set_xlabel("Universal Length Confounder Strength c")
    ax.set_ylabel("Held-Out Item Spearman")
    ax.set_title("D. SummEval Cross-Family Confounder")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.3)

    # Panel E: MC Cross-Family Confounder + Cloned Adversary Recovery
    ax = axes[1, 1]
    mc_r = res_m3["mc_confounder_500"]
    cs_mc = [r["c_confounder"] for r in mc_r]
    ax.plot(cs_mc, [r["CR-RB + DS-EM"] for r in mc_r], "o-", color="#2ca02c", linewidth=2.2, label="CR-RB + DS-EM")
    ax.plot(cs_mc, [r["CR-RB-Spec"] for r in mc_r], "s--", color="#17becf", linewidth=1.9, label="CR-RB-Spec")
    ax.plot(cs_mc, [r["Vanilla RB-Spec"] for r in mc_r], "d--", color="#d62728", linewidth=1.8, label="Vanilla RB-Spec")
    ax.plot(cs_mc, [r["Vanilla DS-EM"] for r in mc_r], "v:", color="#9467bd", linewidth=1.8, label="Vanilla DS-EM")
    ax.plot(cs_mc, [r["Majority Vote"] for r in mc_r], "^:", color="#ff7f0e", linewidth=1.8, label="Majority Vote")
    ax.set_xlabel("Shared Confounder c (500 MC Reps, 42% Adv)")
    ax.set_ylabel("Item Classification Accuracy")
    ax.set_title("E. Monte Carlo Confounder + Adversarial Recovery")
    ax.set_ylim(0.45, 1.01)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.1)

    # Panel F: Empirical Davis-Kahan Spectral Perturbation vs Lean Bound
    ax = axes[1, 2]
    dk_r = res_m4["rb2_dk_curve"]
    ns = np.array([r["n_prompts"] for r in dk_r], dtype=np.float64)
    s2 = np.array([r["mean_sin_sq"] for r in dk_r], dtype=np.float64)
    bd = np.array([r["mean_lean_bound"] for r in dk_r], dtype=np.float64)
    ax.loglog(ns, s2, "o-", color="#2ca02c", linewidth=2.2, label=f"Empirical sin^2(theta) (slope {res_m4['fitted_loglog_slope_sin_sq']:.2f})")
    ax.loglog(ns, bd, "s--", color="#d62728", linewidth=1.9, label=f"Lean Bound 2*eps/gap (slope {res_m4['fitted_loglog_slope_lean_bound']:.2f})")
    ax.set_xlabel("Sample Size n Prompts (log scale)")
    ax.set_ylabel("Subspace Error sin^2(theta)")
    ax.set_title("F. Davis-Kahan Bound vs Empirical Convergence")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.4)

    fig.savefig(RESULTS_DIR / "exp14_calibration_rb2_and_confounders_6panel.png", dpi=300)
    fig.savefig(RESULTS_DIR / "exp14_calibration_rb2_and_confounders_6panel.pdf")
    plt.close(fig)


def write_csvs(res_m1: dict, res_m2: dict, res_m3: dict, res_m4: dict) -> None:
    """Export machine-readable CSV ledgers for all 4 modules."""
    with (RESULTS_DIR / "summeval_calibration_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m_anchors", "item_sp", "sys_sp", "bias", "mae", "rmse", "icc_a1", "icc_a1_lo", "icc_a1_hi", "icc_c1", "unif_icc_a1", "unif_mae"])
        for r in res_m1["splits_200"]:
            w.writerow([r["m_anchors"], f"{r['item_sp']:.4f}", f"{r['sys_sp']:.4f}", f"{r['bias']:.4f}", f"{r['mae']:.4f}", f"{r['rmse']:.4f}", f"{r['icc_a1']:.4f}", f"{r['icc_a1_lo']:.4f}", f"{r['icc_a1_hi']:.4f}", f"{r['icc_c1']:.4f}", f"{r['unif_icc_a1']:.4f}", f"{r['unif_mae']:.4f}"])

    with (RESULTS_DIR / "rewardbench2_regimes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "method", "prompt_acc", "acc_lo", "acc_hi", "macro_acc", "wilcoxon_vs_mean_p", "wilcoxon_vs_svd_p"])
        for reg_label, block in (
            ("natural_23", res_m2["natural_23"]),
            ("weak_clone_flood_32", res_m2["weak_clone_sweep"][-1]["metrics"]),
            ("natural_polarity_flood_51", res_m2["natural_polarity_flood_51"]),
        ):
            for m, d in block.items():
                w.writerow([reg_label, m, f"{d['prompt_acc']:.4f}", f"{d['prompt_acc_ci'][0]:.4f}", f"{d['prompt_acc_ci'][1]:.4f}", f"{d['macro_acc']:.4f}", f"{d['wilcoxon_vs_mean_p']:.3e}", f"{d['wilcoxon_vs_naive_svd_p']:.3e}"])

    with (RESULTS_DIR / "rewardbench2_calibration.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m_anchor_prompts", "rb_ece", "rb_brier", "rb_logloss", "raw_ece", "raw_brier", "raw_logloss", "pair_acc"])
        for r in res_m2["platt_calibration"]:
            w.writerow([r["m_anchor_prompts"], f"{r['ece']:.4f}", f"{r['brier']:.4f}", f"{r['logloss']:.4f}", f"{r['raw_ece']:.4f}", f"{r['raw_brier']:.4f}", f"{r['raw_logloss']:.4f}", f"{r['acc']:.4f}"])

    with (RESULTS_DIR / "confounder_sweep.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "c", "method", "metric"])
        for r in res_m3["summeval_confounder"]:
            for k in ("Uniform Mean", "Vanilla RB-Spec", "CR-RB-Spec", "RB-Spec + EM (5-anc)"):
                w.writerow(["SummEval_item_sp", r["c_confounder"], k, f"{r[k]:.4f}"])
        for r in res_m3["mtbench_confounder"]:
            for k in ("Uniform Mean", "Vanilla RB-Spec", "CR-RB-Spec"):
                w.writerow(["MTBench_acc", r["c_confounder"], k, f"{r[k]:.4f}"])
        for r in res_m3["mc_confounder_500"]:
            for k in ("Majority Vote", "Vanilla DS-EM", "Vanilla RB-Spec", "CR-RB-Spec", "CR-RB + DS-EM"):
                w.writerow(["MC_500_acc", r["c_confounder"], k, f"{r[k]:.4f}"])

    with (RESULTS_DIR / "davis_kahan_curve.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_prompts", "mean_sin_sq", "sin_sq_lo", "sin_sq_hi", "mean_sin", "mean_lean_bound", "bound_violations"])
        for r in res_m4["rb2_dk_curve"]:
            w.writerow([r["n_prompts"], f"{r['mean_sin_sq']:.6f}", f"{r['sin_sq_ci'][0]:.6f}", f"{r['sin_sq_ci'][1]:.6f}", f"{r['mean_sin']:.4f}", f"{r['mean_lean_bound']:.6f}", r["bound_violations"]])


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    print("Loading cached SummEval, MT-Bench, and RewardBench 2 bundles...")
    summeval = load_summeval_bundle()
    mtbench = load_mtbench_bundle()
    rb2 = load_rewardbench2_bundle()

    print("Running Module 1: SummEval m-Anchor Affine Calibration...")
    res_m1 = module1_summeval_calibration(summeval, rng)

    print("Running Module 2: RewardBench 2 Held-Out CV & Platt Calibration...")
    res_m2 = module2_rewardbench2(rb2, rng)

    print("Running Module 3: Universal Confounder Breakdown & CR-RB-Spec/EM Recovery...")
    res_m3 = module3_confounder_breakdown_and_recovery(summeval, mtbench, rb2, rng)

    print("Running Module 4: Empirical Davis-Kahan Spectral Perturbation Curve...")
    res_m4 = module4_davis_kahan_rate(rb2, rng)

    make_figure(res_m1, res_m2, res_m3, res_m4)
    write_csvs(res_m1, res_m2, res_m3, res_m4)

    elapsed = time.perf_counter() - t0
    out = {
        "seed": SEED,
        "elapsed_sec": elapsed,
        "module1_summeval_calibration": res_m1,
        "module2_rewardbench2": res_m2,
        "module3_confounders": res_m3,
        "module4_davis_kahan": res_m4,
    }
    (RESULTS_DIR / "summary_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"DONE Experiment 14 suite in {elapsed:.2f}s!")


if __name__ == "__main__":
    main()
