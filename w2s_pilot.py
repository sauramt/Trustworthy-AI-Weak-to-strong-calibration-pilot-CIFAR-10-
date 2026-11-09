"""Weak-to-strong (W2S) calibration pilot: is a weakly supervised strong student still calibrated?

Purpose : Burns et al. (2024) evaluate weak-to-strong generalisation with accuracy and PGR only.
          Their auxiliary confidence loss (App. A.4, eq. 1) explicitly pushes the student toward its
          own hardened predictions. This pilot measures what the paper does not: calibration (ECE,
          NLL, Brier), whether the student's uncertainty flags the supervisor errors it imitates, and
          what happens to conformal prediction sets when the only labels available for calibration
          are the weak ones. It is a pilot for the course's midterm talk (critique slide) and for the
          A3 mini-project proposal, not a result for the final paper.
Setting : Paper's vision setup (App. "Self-supervised vision models") on CIFAR-10:
            weak supervisor = linear probe on frozen AlexNet fc7 features, trained on ground truth;
            strong student  = linear probe on frozen DINO ViT-B/8 CLS features, trained on the weak
                              supervisor's soft labels (naive) or with the confidence loss;
            strong ceiling  = the same strong probe trained on ground truth.
          Probe training follows the paper: Adam, lr 1e-3, batch 128, no weight decay.
Inputs  : ./features/*.npy written by extract_features.py
Outputs : ./results/metrics.json  (every number quoted anywhere comes from here)
          ./results/curves.csv     (test accuracy per epoch per method/seed: the overfitting check)
          ./results/test_probs.npz (test probabilities of every model, for local figures)
Run     : python w2s_pilot.py     (47 s on an RTX 5060 laptop GPU; CPU also works, a few minutes per
                                  probe, not timed)
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FEAT, OUT = HERE / "features", HERE / "results"
device = "cuda" if torch.cuda.is_available() else "cpu"
K = 10                       # CIFAR-10 classes
EPOCHS = 20                  # fixed budget (the paper's ViT-B/8 ImageNet probe used 5; we log every epoch)
N_BOOT = 5                   # bootstrap members per method (A3's ensemble idea, applied to the probe)
ALPHA_MAX = 0.75             # confidence-loss weight for the largest students (App. A.4)
WARMUP = 0.2                 # alpha warms up linearly over the first 20 % of steps (App. A.4)
CONF_LEVEL = 0.90            # conformal target coverage (1 - alpha_conformal)

# ---------------------------------------------------------------- data and splits
def load(name, split):
    """Load cached features (float16 on disk) as float32 tensors."""
    return torch.tensor(np.load(FEAT / f"{name}_{split}.npy").astype(np.float32))

Xw_tr, Xw_te = load("alexnet", "train"), load("alexnet", "test")
Xs_tr, Xs_te = load("dino", "train"), load("dino", "test")
y_tr = torch.tensor(np.load(FEAT / "labels_train.npy")).long()
y_te = torch.tensor(np.load(FEAT / "labels_test.npy")).long()

# Standardise each representation with statistics of the CIFAR training split (no test leakage).
for a, b in [(Xw_tr, Xw_te), (Xs_tr, Xs_te)]:
    mu, sd = a.mean(0), a.std(0) + 1e-6
    a.sub_(mu).div_(sd); b.sub_(mu).div_(sd)

# Disjoint splits of the 50k CIFAR training images, as in the paper's protocol (weak supervisor and
# student never see the same images):  20k to train the weak supervisor on ground truth,
# 25k to train the student (on weak labels, or on ground truth for the ceiling),
# 5k held out for conformal calibration. Test = the official 10k test split.
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(y_tr), generator=g)
i_weak, i_w2s, i_cal = perm[:20_000], perm[20_000:45_000], perm[45_000:]

# ---------------------------------------------------------------- probe training
def train_probe(X, target, loss="soft_ce", seed=0, boot=False, X_eval=None, y_eval=None, log=None):
    """Train a linear probe (logistic regression in torch) and return it.

    X       : features of the training images
    target  : either hard labels (LongTensor) or soft labels (FloatTensor, rows sum to 1)
    loss    : "soft_ce" = cross-entropy to the target distribution (naive W2S / GT training);
              "conf"    = the auxiliary confidence loss of Burns et al. (App. A.4, eq. 2 form):
                          CE(f(x), (1-alpha) f_w(x) + alpha * onehot(argmax f(x))).
              For 10 classes the paper's adaptive binary threshold becomes an argmax; the paper's
              footnote says the prior should be set from the weak predictions in non-binary
              settings, which we skip in this pilot (stated as a limitation).
    boot    : True = train on a bootstrap resample (with replacement) for the ensemble.
    X_eval, y_eval, log : if given, ground-truth test accuracy is logged after every epoch (eval only;
              it never influences training, so this is not ground-truth early stopping).
    """
    torch.manual_seed(seed)
    n = len(X)
    idx = torch.randint(0, n, (n,), generator=torch.Generator().manual_seed(seed)) if boot else torch.arange(n)
    X, target = X[idx].to(device), target[idx].to(device)
    if target.dtype == torch.long:                       # hard labels -> one-hot distribution
        target = F.one_hot(target, K).float()
    probe = torch.nn.Linear(X.shape[1], K).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)  # paper: Adam, 1e-3, no weight decay
    steps_per_epoch = (n + 127) // 128
    total, step = EPOCHS * steps_per_epoch, 0
    for ep in range(EPOCHS):
        order = torch.randperm(n, device=device)
        for s in range(0, n, 128):                       # paper: batch size 128
            b = order[s:s + 128]
            logits = probe(X[b])
            tgt = target[b]
            if loss == "conf":
                alpha = ALPHA_MAX * min(1.0, step / (WARMUP * total))     # linear warm-up
                hard = F.one_hot(logits.argmax(1), K).float().detach()    # hardened own prediction
                tgt = (1 - alpha) * tgt + alpha * hard
            l = -(tgt * F.log_softmax(logits, 1)).sum(1).mean()           # soft cross-entropy
            opt.zero_grad(); l.backward(); opt.step(); step += 1
        if log is not None:
            log.append((ep + 1, accuracy(predict(probe, X_eval), y_eval)))
    return probe


@torch.no_grad()
def predict(probe, X):
    """Softmax probabilities of a probe on features X (returned on CPU)."""
    return F.softmax(probe(X.to(device)), 1).cpu()


def accuracy(p, y):
    return (p.argmax(1) == y).float().mean().item()

# ---------------------------------------------------------------- metrics
def ece(p, y, bins=15):
    """Top-label expected calibration error with equal-width confidence bins (Guo et al., 2017)."""
    conf, pred = p.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(correct[m].mean().item() - conf[m].mean().item())
    return e


def nll(p, y):
    return -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)).mean().item()


def brier(p, y):
    return ((p - F.one_hot(y, K).float()) ** 2).sum(1).mean().item()


def auroc(score, positive):
    """AUROC of `score` for detecting `positive` (Mann-Whitney U, ties averaged)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    order = score.argsort()
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = (sums / cnt)[inv]
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def entropy(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(1)


def conformal(p_cal, y_cal, p_test, y_test_true, level=CONF_LEVEL):
    """Split conformal (LAC score 1 - p_y; Sadinle et al., 2019). Returns coverage of the TRUE
    test labels and mean set size. y_cal may be weak labels: that is the point of the test."""
    s = 1 - p_cal[torch.arange(len(y_cal)), y_cal]
    n = len(s)
    q = torch.sort(s).values[min(n - 1, int(np.ceil((n + 1) * level)) - 1)].item()
    sets = (1 - p_test) <= q
    return {"qhat": q,
            "coverage_true": sets[torch.arange(len(y_test_true)), y_test_true].float().mean().item(),
            "mean_set_size": sets.sum(1).float().mean().item()}


def summarise(p, y):
    return {"acc": accuracy(p, y), "ece": ece(p, y), "nll": nll(p, y), "brier": brier(p, y),
            "mean_conf": p.max(1).values.mean().item()}

# ---------------------------------------------------------------- run
t0 = time.time()
R = {"setting": {"weak": "AlexNet fc7 linear probe", "strong": "DINO ViT-B/8 CLS linear probe",
                 "data": "CIFAR-10", "n_weak_train": len(i_weak), "n_w2s_train": len(i_w2s),
                 "n_cal": len(i_cal), "n_test": len(y_te), "epochs": EPOCHS, "n_boot": N_BOOT,
                 "alpha_max": ALPHA_MAX, "conformal_level": CONF_LEVEL, "device": device}}
curves = []

# 1) Weak supervisor: AlexNet probe trained on ground truth of the 20k weak split.
weak = train_probe(Xw_tr[i_weak], y_tr[i_weak], seed=0)
pw_te = predict(weak, Xw_te)
pw_w2s = predict(weak, Xw_tr[i_w2s])          # soft weak labels for the student's training images
pw_cal = predict(weak, Xw_tr[i_cal])          # weak labels for the calibration images
R["weak"] = summarise(pw_te, y_te)
R["weak"]["weak_label_acc_on_w2s_split"] = accuracy(pw_w2s, y_tr[i_w2s])
print(f"weak {R['weak']['acc']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# 2) Ceiling, naive W2S and confidence-loss W2S, each as N_BOOT bootstrap members (+ one full-data fit).
methods = {"ceiling": (y_tr[i_w2s], "soft_ce"), "naive": (pw_w2s, "soft_ce"), "conf": (pw_w2s, "conf")}
P = {"weak": pw_te.numpy()}
for m, (target, loss) in methods.items():
    log = []
    full = train_probe(Xs_tr[i_w2s], target, loss=loss, seed=0, X_eval=Xs_te, y_eval=y_te, log=log)
    curves += [(m, "full", ep, a) for ep, a in log]
    p_full, p_cal = predict(full, Xs_te), predict(full, Xs_tr[i_cal])
    members = [predict(train_probe(Xs_tr[i_w2s], target, loss=loss, seed=100 + b, boot=True), Xs_te)
               for b in range(N_BOOT)]
    p_ens = torch.stack(members).mean(0)
    mi = entropy(p_ens) - torch.stack([entropy(q) for q in members]).mean(0)   # epistemic part (A3)
    r = summarise(p_full, y_te)
    r["ensemble"] = summarise(p_ens, y_te)
    r["member_acc"] = [accuracy(q, y_te) for q in members]

    pred, wpred = p_full.argmax(1), pw_te.argmax(1)
    wrong, weak_wrong = pred != y_te, wpred != y_te
    imitated = wrong & (pred == wpred)                  # student repeats the supervisor's exact error
    corrected = (~wrong) & weak_wrong                   # student fixes a supervisor error
    conf = p_full.max(1).values
    r["share_errors_imitated"] = (imitated.sum() / wrong.sum()).item()
    r["n_corrected"] = int(corrected.sum()); r["n_imitated"] = int(imitated.sum())
    r["n_weak_errors"] = int(weak_wrong.sum())
    r["mean_conf_correct"] = conf[~wrong].mean().item()
    r["mean_conf_imitated_errors"] = conf[imitated].mean().item() if imitated.any() else float("nan")
    r["mean_conf_other_errors"] = conf[wrong & ~imitated].mean().item() if (wrong & ~imitated).any() else float("nan")
    # Label-free error detection (higher score = more likely wrong), as in A3 Part 2.
    r["auroc_err_maxprob"] = auroc(-conf.numpy(), wrong.numpy())
    r["auroc_err_entropy"] = auroc(entropy(p_full).numpy(), wrong.numpy())
    r["auroc_err_ens_mi"] = auroc(mi.numpy(), (p_ens.argmax(1) != y_te).numpy())
    # Can the student's confidence tell its imitated errors apart from its correct answers?
    sub = (~wrong) | imitated
    r["auroc_imitated_vs_correct_maxprob"] = auroc(-conf[sub].numpy(), imitated[sub].numpy())
    # Conformal: calibrate with weak labels (all a W2S practitioner has) vs ground truth (oracle).
    r["conformal_weak_cal"] = conformal(p_cal, pw_cal.argmax(1), p_full, y_te)
    r["conformal_true_cal"] = conformal(p_cal, y_tr[i_cal], p_full, y_te)
    R[m] = r
    P[m] = p_full.numpy(); P[m + "_ens"] = p_ens.numpy()
    print(f"{m} acc {r['acc']:.4f} ece {r['ece']:.4f} ({time.time()-t0:.0f}s)", flush=True)

# 3) PGR = (W2S - weak) / (ceiling - weak), on accuracy (paper §3).
for m in ("naive", "conf"):
    R[m]["pgr"] = (R[m]["acc"] - R["weak"]["acc"]) / (R["ceiling"]["acc"] - R["weak"]["acc"])
    R[m]["pgr_best_epoch"] = (max(a for mm, _, _, a in curves if mm == m) - R["weak"]["acc"]) / \
                             (R["ceiling"]["acc"] - R["weak"]["acc"])
R["runtime_s"] = time.time() - t0

OUT.mkdir(exist_ok=True)
(OUT / "metrics.json").write_text(json.dumps(R, indent=2))
with open(OUT / "curves.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "fit", "epoch", "test_acc"]); w.writerows(curves)
np.savez_compressed(OUT / "test_probs.npz", y=y_te.numpy(), **P)
print("done", json.dumps({m: {k: round(R[m][k], 4) for k in ("acc", "ece", "nll")} for m in
                          ("weak", "ceiling", "naive", "conf")}), flush=True)
