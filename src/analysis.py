"""Pre-registered analyses (PREREGISTRATION.md) on out-of-fold recording-level predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 1000


# ---------------------------------------------------------------- loading
def load(root: Path = ROOT):
    """Recordings, species table and {model: probs}. All arrays share the recording order of 03."""
    bl = root / "cache/biolingual"
    rec = pd.read_csv(bl / "recordings.csv")
    classes = json.loads((bl / "classes.json").read_text())
    sp = pd.read_csv(root / "outputs/eda/species.csv", index_col=0).loc[classes]
    sp["rec_per_author"] = sp["n_rec"] / sp["n_author"]
    sp["inat_share"] = sp["n_iNat"] / sp["n_rec"]
    sp["genus"] = sp["scientific_name"].str.split().str[0]

    probs = {f"M3_{p}": np.load(bl / f"probs_m3_{p}.npy") for p in ["P1", "P2", "P3"]}
    probs["M4"] = np.load(bl / "probs_m4.npy")
    cnn = root / "cache/cnn"
    if (cnn / "probs_m2.npy").exists():
        rec2 = pd.read_csv(cnn / "recordings_m2.csv")
        assert (rec2["filename"].to_numpy() == rec["filename"].to_numpy()).all(), "M2 recording order differs"
        assert json.loads((cnn / "classes.json").read_text()) == classes
        probs["M2"] = np.load(cnn / "probs_m2.npy")
    for k, P in probs.items():
        assert P.shape == (len(rec), len(classes)), k
    return rec, sp, probs


# ---------------------------------------------------------------- metrics
def per_species(y, pred, n_classes):
    """Recall, precision, F1 per class from top-1 predictions."""
    tp = np.bincount(y[pred == y], minlength=n_classes).astype(float)
    support = np.bincount(y, minlength=n_classes).astype(float)
    predicted = np.bincount(pred, minlength=n_classes).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = tp / support
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({"recall": recall, "precision": precision, "f1": f1, "support": support})


def summary(rec, probs):
    y = rec["label_idx"].to_numpy()
    rows = []
    for name, P in probs.items():
        pred = P.argmax(1)
        m = per_species(y, pred, P.shape[1])
        top5 = (np.argsort(-P, axis=1)[:, :5] == y[:, None]).any(1).mean()
        rows.append(dict(model=name, accuracy=(pred == y).mean(), macro_f1=m["f1"].mean(),
                         macro_recall=m["recall"].mean(), top5=top5))
    return pd.DataFrame(rows).set_index("model")


def species_bootstrap_ci(rec, P, n_boot=N_BOOT, seed=0):
    """95% CI of per-species recall and F1: resample recordings within each species (stratified)."""
    rng = np.random.default_rng(seed)
    y = rec["label_idx"].to_numpy()
    pred = P.argmax(1)
    groups = [np.flatnonzero(y == c) for c in range(P.shape[1])]
    rec_b, f1_b = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        m = per_species(y[idx], pred[idx], P.shape[1])
        rec_b.append(m["recall"].to_numpy()); f1_b.append(m["f1"].to_numpy())
    q = lambda a: np.percentile(np.array(a), [2.5, 97.5], axis=0)
    (rl, rh), (fl, fh) = q(rec_b), q(f1_b)
    return pd.DataFrame({"recall_lo": rl, "recall_hi": rh, "f1_lo": fl, "f1_hi": fh})


# ---------------------------------------------------------------- regression helpers
def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def slope_test(x, y, covariates=None, n_boot=N_BOOT, seed=0):
    """Spearman rho and OLS slope of y on x (+ covariates), with 95% bootstrap CIs over species."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = None if covariates is None else np.asarray(covariates, float)
    design = lambda i: x[i] if C is None else np.column_stack([x[i], C[i]])
    allidx = np.arange(len(x))
    rho = spearmanr(x, y)[0]
    b = ols(y, design(allidx))[1]
    rng = np.random.default_rng(seed)
    rhos, bs = [], []
    for _ in range(n_boot):
        i = rng.choice(allidx, len(allidx))
        rhos.append(spearmanr(x[i], y[i])[0]); bs.append(ols(y[i], design(i))[1])
    ci = lambda a: tuple(np.nanpercentile(a, [2.5, 97.5]))
    return dict(n_species=len(x), rho=rho, rho_ci=ci(rhos), slope=b, slope_ci=ci(bs),
                _boot_slopes=np.array(bs))


def zscore(df):
    return (df - df.mean()) / df.std()


def fmt(res, label):
    return (f"{label}: n={res['n_species']}  rho={res['rho']:.3f} [{res['rho_ci'][0]:.3f}, {res['rho_ci'][1]:.3f}]  "
            f"slope={res['slope']:.4f} [{res['slope_ci'][0]:.4f}, {res['slope_ci'][1]:.4f}]  "
            f"supported={'yes' if res['slope_ci'][0] > 0 else 'no'}")


# ---------------------------------------------------------------- hypotheses
H1_COVARIATES = ["rec_per_author", "mean_duration", "mean_rating", "inat_share"]


def h1(rec, sp, P, mask=None):
    """M2 per-species F1 ~ log(n_rec) + covariates (z-scored)."""
    r, P = (rec, P) if mask is None else (rec[mask].reset_index(drop=True), P[mask])
    m = per_species(r["label_idx"].to_numpy(), P.argmax(1), P.shape[1])
    ok = m["support"].to_numpy() > 0
    x = np.log(sp["n_rec"].to_numpy())[ok]
    simple = slope_test(x, m["f1"].to_numpy()[ok])
    adjusted = slope_test(x, m["f1"].to_numpy()[ok], zscore(sp[H1_COVARIATES]).to_numpy()[ok])
    return simple, adjusted


def recall_vs_exposure(rec, sp, P, subset=None, min_support=1):
    """Per-species recall on (a subset of) recordings vs log(AnimalSpeak captions)."""
    r, Q = (rec, P) if subset is None else (rec[subset].reset_index(drop=True), P[subset])
    m = per_species(r["label_idx"].to_numpy(), Q.argmax(1), Q.shape[1])
    ok = m["support"].to_numpy() >= min_support
    return np.log(sp["as_n_total"].to_numpy())[ok], m["recall"].to_numpy()[ok]


def h2(rec, sp, P):
    x, y = recall_vs_exposure(rec, sp, P)
    xu, yu = recall_vs_exposure(rec, sp, P, subset=(rec["leak"] == "unseen").to_numpy(), min_support=5)
    return slope_test(x, y), slope_test(xu, yu)


def h3(rec, sp, P_zero, P_probe, subset=None, min_support=1, n_boot=N_BOOT, seed=0):
    """slope(M3) - slope(M4) of recall on log(AnimalSpeak), paired bootstrap over species."""
    x, a = recall_vs_exposure(rec, sp, P_zero, subset, min_support)
    _, b = recall_vs_exposure(rec, sp, P_probe, subset, min_support)
    diff = lambda i: ols(a[i], x[i])[1] - ols(b[i], x[i])[1]
    allidx = np.arange(len(x))
    rng = np.random.default_rng(seed)
    boots = [diff(rng.choice(allidx, len(allidx))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_species=len(x), slope_zero=ols(a, x)[1], slope_probe=ols(b, x)[1], diff=diff(allidx),
                diff_ci=(lo, hi), supported=lo > 0)


def h4(rec, sp, P, ks=range(1, 21)):
    """Detection probability 1 - (1 - recall)^k per tertile (mean over species)."""
    m = per_species(rec["label_idx"].to_numpy(), P.argmax(1), P.shape[1])
    m["tertile"] = sp["tertile"].to_numpy()
    rows = []
    for t, g in m.groupby("tertile"):
        for k in ks:
            p = 1 - (1 - g["recall"].to_numpy()) ** k
            rows.append(dict(tertile=t, k=k, mean=p.mean(), min=p.min(), q25=np.percentile(p, 25)))
    return pd.DataFrame(rows), m


# ---------------------------------------------------------------- exploratory
def prompt_sensitivity(rec, sp, P1, P2):
    y = rec["label_idx"].to_numpy()
    r1 = per_species(y, P1.argmax(1), P1.shape[1])["recall"].to_numpy()
    r2 = per_species(y, P2.argmax(1), P2.shape[1])["recall"].to_numpy()
    return slope_test(np.log(sp["as_n_total"].to_numpy()), np.abs(r1 - r2))


def confusion_flow(rec, sp, P):
    """Where errors of each tertile go: share to high-tertile classes and to same-genus classes."""
    y = rec["label_idx"].to_numpy(); pred = P.argmax(1)
    tert = sp["tertile"].to_numpy(); genus = sp["genus"].to_numpy()
    wrong = pred != y
    out = []
    for t in ["low", "mid", "high"]:
        sel = wrong & (tert[y] == t)
        # chance baseline: share of high-tertile classes among the other 144 classes
        others_high = np.mean([(tert[np.arange(len(tert)) != c] == "high").mean() for c in np.flatnonzero(tert == t)])
        out.append(dict(true_tertile=t, n_errors=int(sel.sum()),
                        to_high=(tert[pred[sel]] == "high").mean(), to_high_chance=others_high,
                        same_genus=(genus[pred[sel]] == genus[y[sel]]).mean()))
    return pd.DataFrame(out).set_index("true_tertile")
