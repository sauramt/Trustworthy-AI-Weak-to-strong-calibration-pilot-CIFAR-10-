# From this pilot to our group project

Our discussion deck (Group 3, 11 Nov) plans: weak supervisor → strong student on CIFAR-10, shift with
CIFAR-10-C noise and blur, uncertainty from softmax / MC Dropout / ensembles, abstention curves, and
possibly an uncertainty-weighted confidence loss and conformal sets. This page says what the pilot
already gives, what each extension needs in the code, and roughly what it costs. Nothing here is decided:
the scope is for us and the professor to settle on Wednesday.

## What already exists

| Plan item | Status in this code |
|---|---|
| Weak supervisor, naive student, confidence-loss student, ceiling | done (`w2s_pilot.py`) |
| PGR, accuracy, ECE, NLL, Brier, reliability diagrams | done |
| Softmax uncertainty (max-prob, entropy) and error-detection AUROC | done |
| Ensemble (5 bootstrap members per method) + mutual information | done |
| Copied-supervisor-error analysis | done |
| Conformal sets with weak vs true calibration labels | done (5k calibration split `i_cal`) |
| A small clean calibration set (our feedback question 3) | the 5k `i_cal` images have true labels; the code already uses them as the oracle |
| Accuracy–coverage (abstention) curves | **not in the scripts yet, but computable now on a CPU** (below) |
| CIFAR-10-C shift | not done |
| MC Dropout | not done (a linear probe has no dropout) |
| Uncertainty-weighted confidence loss | not done |

## 1. Abstention curves: possible today, no GPU

`results/test_probs.npz` holds every model's test probabilities. Rejecting the least confident inputs
and measuring accuracy on the rest:

```python
import numpy as np
P = np.load("results/test_probs.npz")
y = P["y"]
for m in ["weak", "naive", "conf", "ceiling"]:
    p = P[m]
    conf, correct = p.max(1), p.argmax(1) == y
    order = np.argsort(-conf)                                       # most confident first
    acc_at = np.cumsum(correct[order]) / np.arange(1, len(y) + 1)   # accuracy when keeping the top k
    aurc = np.mean(1 - acc_at)                                      # area under the risk-coverage curve
    print(m, {c: round(acc_at[int(c * len(y)) - 1], 4) for c in (1.0, 0.9, 0.8, 0.5)}, round(aurc, 4))
```

Output on the shipped results (clean test set, run 9 Nov 2026):

| Model | Accuracy, keep 100 % | keep 90 % | keep 80 % | keep 50 % | AURC (lower is better) |
|---|---|---|---|---|---|
| Weak supervisor | 79.8 % | 83.9 % | 87.8 % | 95.8 % | 0.065 |
| Student, naive | 88.8 % | 92.8 % | 95.9 % | 99.0 % | 0.022 |
| Student, confidence loss | 94.0 % | 97.5 % | 98.9 % | 99.8 % | 0.008 |
| Ceiling | 96.7 % | 99.3 % | 99.8 % | 100.0 % | 0.002 |

Replace `p.max(1)` by any other uncertainty score (entropy, ensemble MI, MC Dropout variance) to compare
signals. The same code runs on corrupted-test probabilities once they exist.

## 2. Distribution shift with CIFAR-10-C

CIFAR-10-C (Hendrycks & Dietterich, 2019) gives the 10k CIFAR-10 test images under 19 corruptions at 5
severities. Each corruption is one `.npy` file of shape (50000, 32, 32, 3), uint8, severity 1 to 5 in
blocks of 10k, plus a shared `labels.npy`. Noise: `gaussian_noise`, `shot_noise`, `impulse_noise`;
blur: `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`. Download it from the dataset's Zenodo
page (linked from github.com/hendrycks/robustness).

What changes in the code:

- **Only test features change.** The probes are trained on clean features exactly as now. Add a script
  (e.g. `extract_features_c.py`) that wraps the `.npy` arrays in a `TensorDataset`, applies the same
  transform (to PIL → resize 224 → normalise) and saves `features/{alexnet,dino}_c_<corruption>.npy`.
- `w2s_pilot.py` does **not save the trained probes**. Either save them
  (`torch.save(probe.state_dict(), ...)` after each `train_probe`) and evaluate later, or evaluate the
  corrupted features inside the same loop where `p_full` is computed, and add the results to `R[m]`.
- **Copied errors under shift** need the *weak supervisor's* predictions on the corrupted images too,
  which is why the AlexNet features of the corrupted images are needed, not only the DINO ones.

Compute estimate (not measured): the original extraction pushed 60k images through both backbones in
about 15 min on an 8 GB laptop GPU. Two corruptions × 5 severities = 100k images, so roughly 25 min on
similar hardware. All 19 corruptions would be roughly 4 h.

## 3. MC Dropout

The student is a single `nn.Linear` on frozen features, so there is no dropout to switch on. Two cheap
options:

- dropout on the input features: `nn.Sequential(nn.Dropout(p), nn.Linear(768, 10))`, trained as now, and
  at test time `model.train()` + T ≈ 20–50 stochastic passes, averaging the softmax;
- a one-hidden-layer head with dropout: `Linear(768, 512) → ReLU → Dropout(p) → Linear(512, 10)`.

Either takes seconds on cached features. Note in the report that this is dropout in the **head only**,
not in the frozen backbone, so it is a weaker form of MC Dropout than in Gal & Ghahramani (2016).

## 4. Ensembles

`train_probe(..., boot=True, seed=100 + b)` already trains bootstrap members. A "5-head ensemble" of the
deep-ensembles kind (different seeds, full data, no bootstrap) is the same call with `boot=False` and
different seeds. Cost: seconds per member.

## 5. Uncertainty-weighted confidence loss (if we keep it)

In `train_probe`, the confidence loss builds its target here:

```python
alpha = ALPHA_MAX * min(1.0, step / (WARMUP * total))     # linear warm-up
hard = F.one_hot(logits.argmax(1), K).float().detach()    # hardened own prediction = f^_t in the paper
tgt = (1 - alpha) * tgt + alpha * hard
```

In feedback on my A02 proposal, the professor suggested focusing on this hardened prediction f̂_t (rather than
on α). Possible variants, each a few lines here: harden only when the student's confidence exceeds a
threshold; replace the one-hot by a temperature-sharpened distribution; weight α per example by the
student's or the supervisor's entropy. The paper's balanced adaptive threshold (App. A.4) is the
baseline we replaced by an argmax.

## 6. Conformal prediction (if we keep it)

Already implemented (`conformal()`, LAC score). Under shift, the interesting question is whether a set
calibrated on clean weak labels still covers the truth on corrupted images.

## References for this page

- Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions
  and perturbations. *ICLR 2019*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty
  in deep learning. *ICML 2016*, PMLR 48, 1050–1059.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty
  estimation using deep ensembles. *NeurIPS 2017*.
- Burns et al. (2024), see the README.
