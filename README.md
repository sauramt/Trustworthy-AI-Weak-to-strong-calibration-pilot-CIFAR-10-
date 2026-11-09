# Weak-to-strong calibration pilot (CIFAR-10)

Group 3, Trustworthy AI (NYCU 536904), Fall 2026. Pilot by Tomás Saura Martínez, run 29 Oct 2026.

This folder holds the experiment behind the numbers on slide 2 of our project-discussion deck. It is a
small reproduction of the vision experiment in Burns et al. (2024), *Weak-to-Strong Generalization*
(the paper I presented, A02), extended with **uncertainty measurements the paper does not report**.
The paper judges a weakly supervised student by accuracy alone. This pilot asks whether the student is
also **calibrated**, whether its uncertainty **flags the supervisor errors it copies**, and what happens to
**conformal prediction sets** when the only labels available for calibration are the weak ones.

It is a **pilot, not a result**: one dataset, one supervisor–student pair, linear probes, one seed for the
main fit, no hyper-parameter search. It was built to check that the idea is feasible for a final project.

The code was written with an AI assistant (Claude) under my direction. I reviewed it and ran it.

---

## 1. The setting in one picture

```
 CIFAR-10 train (50k), split once with seed 0, disjoint parts
 ├── 20k  ──► WEAK SUPERVISOR: linear probe on frozen AlexNet features, trained on TRUE labels
 ├── 25k  ──► the supervisor labels these images (soft probabilities = "weak labels")
 │            ├─► STRONG STUDENT, naive:            linear probe on frozen DINO ViT-B/8 features, trained on weak labels
 │            ├─► STRONG STUDENT, confidence loss:  same, with the paper's auxiliary confidence loss
 │            └─► STRONG CEILING:                   same probe trained on the TRUE labels of these 25k images
 └──  5k  ──► calibration set for conformal prediction (used with weak labels, and with true labels as an oracle)
 CIFAR-10 test (10k) ──► every number below
```

Both backbones stay frozen. Only the linear probes (one `nn.Linear` layer each) are trained, which is
what the paper itself does in its vision experiment and recommends for fast iteration.

| Role | Model | Trained on |
|---|---|---|
| Weak supervisor | AlexNet (ImageNet weights, torchvision), fc7 features (4096-d) + linear probe | 20k true labels |
| Strong student (naive) | DINO ViT-B/8 (self-supervised, `facebookresearch/dino`), CLS features (768-d) + linear probe | 25k weak soft labels, cross-entropy |
| Strong student (confidence loss) | same | 25k weak soft labels, auxiliary confidence loss (paper App. A.4): target = (1−α)·weak label + α·hardened own prediction, α warms up to 0.75 over the first 20 % of steps |
| Strong ceiling | same | 25k true labels |

Training follows the paper: Adam, lr 1e-3, batch 128, no weight decay, 20 epochs, no early stopping.
Each method is also trained 5 more times on bootstrap resamples to form a small ensemble.

## 2. Results (10k CIFAR-10 test images)

All numbers come from `results/metrics.json`; `results/pilot_table.csv` has the same in one table.

| Model | Accuracy | PGR | ECE | Mean confidence | Error-detection AUROC (max-prob) |
|---|---|---|---|---|---|
| Weak supervisor (AlexNet probe) | 79.8 % | – | 12.0 % (over-confident) | 0.92 | – |
| Student, naive | 88.8 % | **53 %** | 8.3 % (**under**-confident) | 0.81 | 0.88 |
| Student, confidence loss | 94.0 % | **84 %** | 2.7 % | 0.91 | 0.91 |
| Ceiling (true labels) | 96.7 % | – | 1.5 % | 0.98 | 0.95 |

PGR (performance gap recovered) = (student − weak) / (ceiling − weak), on accuracy.

**What the pilot shows**

1. **The paper's accuracy story reproduces.** Naive training recovers about half the gap (the paper's
   ImageNet table reports 42–58 %); the confidence loss recovers much more. The naive student also
   overfits the weak labels: 92.8 % after epoch 1, 88.8 % after epoch 20 (best-epoch PGR 77 %).
   See `results/fig_pilot_curves.png`.
2. **Calibration moves, and not in the direction I first expected.** The naive student is
   *under*-confident: it imitates the supervisor's spread-out soft labels while its own features are
   better, so it is right more often than its probabilities say. The confidence loss fixes most of that.
   See `results/fig_pilot_reliability.png` (bars above the diagonal = under-confident).
3. **Copied errors are held with more confidence.** 396 of the naive student's 1,120 errors (35 %) and 242
   of the confidence-loss student's 598 errors (40 %) repeat the supervisor's exact mistake. Mean confidence
   on these copied errors vs the student's other errors: naive 0.60 vs 0.52, confidence loss 0.71 vs 0.61.
   Separating copied errors from correct answers by confidence gives AUROC 0.84 (naive) and 0.88 (conf.).
4. **Conformal sets calibrated on weak labels over-cover.** Target 90 %. Calibrated with weak labels (all a
   practitioner has), the sets cover the true label 97.6 % (naive) and 99.3 % (conf.) of the time, with
   1.51 and 1.53 classes on average. Calibrated with true labels: 90.1 % / 89.3 % coverage, 1.03 / 0.92
   classes. So weak-label calibration is safe but wasteful here (≈ 66 % larger sets for the conf. student):
   where the student correctly disagrees with a wrong weak label, the calibration score is huge and the
   threshold is inflated.

**How to quote these numbers.** "ECE 8.3 %" belongs to the **naive** student and means under-confidence.
The 84 % student (confidence loss) has ECE 2.7 %. "66 % larger sets" is the confidence-loss student
(1.53 vs 0.92 classes). Always say "pilot".

Ensemble of the 5 bootstrap members (not better here): naive 89.4 % acc., ECE 8.8 %; confidence loss
94.0 % acc., ECE 3.9 %.

## 3. Folder contents

```
README.md                 this file
requirements.txt          Python packages (install torch/torchvision first, see §4)
run.sh                    runs the three scripts in order
extract_features.py       step 1 (GPU): CIFAR-10 → frozen AlexNet and DINO features in ./features/
w2s_pilot.py              step 2 (GPU or CPU): trains all probes, computes every metric → results/
make_figures.py           step 3 (CPU): figures + pilot_table.csv from results/
run_on_colab.ipynb        the same three steps on Google Colab (free GPU)
results/                  the outputs of the original run (you can make figures without re-running anything)
  metrics.json            every number, per model (see docs/RESULTS_EXPLAINED.md for each key)
  pilot_table.csv         one row per model
  curves.csv              ground-truth test accuracy after each epoch (overfitting check)
  test_probs.npz          test-set probabilities of every model + true labels (2.6 MB): enough to
                          compute new uncertainty metrics or abstention curves on a CPU, no GPU needed
  features_meta.json      shapes of the extracted features, torch version, device
  fig_pilot_*.png         reliability diagrams, accuracy-vs-ECE summary, training curves
docs/
  RESULTS_EXPLAINED.md    every metric and every key of metrics.json, in plain words
  EXTENDING_FOR_THE_GROUP.md  how our group plan (corruptions, abstention, MC Dropout, ensembles,
                          uncertainty-weighted loss) maps onto this code, with compute estimates
context/
  midterm_talk_slides_Burns2024.pdf   my 5-minute talk on the paper (4 Nov), for background
```

Not included (regenerated on the first run): `data/` (CIFAR-10) and `features/` (~0.6 GB).

## 4. Setup

You need about **2 GB of disk** and **internet on the first run**: CIFAR-10 (~170 MB), AlexNet weights
(~230 MB) and DINO ViT-B/8 through `torch.hub` from GitHub (~330 MB) are downloaded automatically.

### Option A: Google Colab (no local GPU needed)

1. Zip this folder (or use the zip I shared) and open `run_on_colab.ipynb` in Colab.
2. Runtime → Change runtime type → **GPU**.
3. Run the cells in order. The notebook uploads the zip, installs nothing extra (Colab has torch), runs
   the three steps and offers `results.zip` for download.

I have not run it on Colab myself (my run was on a local GPU), so tell me if a cell fails.

### Option B: your own machine with an NVIDIA GPU (Linux or Windows)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
# 1) torch + torchvision for YOUR CUDA version: copy the command from https://pytorch.org/get-started/locally/
#    e.g. pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) the rest
pip install numpy matplotlib
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
bash run.sh                            # Windows without bash: run the three python commands in run.sh
```

### Option C: CPU only

`python make_figures.py` works anywhere with numpy and matplotlib, from the shipped `results/`.
`w2s_pilot.py` also runs on CPU once `features/` exists. Feature extraction on CPU is possible but takes
many hours (60k images through a ViT-B/8 at 224 px); use Colab instead.

### Original environment (for reference)

Ubuntu laptop, NVIDIA RTX 5060 Laptop GPU (8 GB), Python 3 venv with torch 2.14.1+cu130, torchvision,
numpy, matplotlib. Feature extraction took about 15 min, all probes 47 s.

## 5. Run

```bash
bash run.sh
```

or step by step:

```bash
python extract_features.py   # writes features/{alexnet,dino}_{train,test}.npy, labels_{train,test}.npy
python w2s_pilot.py          # writes results/metrics.json, curves.csv, test_probs.npz
python make_figures.py       # writes results/fig_pilot_*.png, pilot_table.csv
```

Feature extraction resumes: splits already saved in `features/` are skipped. Re-running `w2s_pilot.py`
overwrites `results/`; copy the folder first if you want to keep the original numbers.

**Reproducibility.** Splits and probe initialisations are seeded. GPU kernels and mixed precision are not
bit-exact, so expect small differences (fractions of a percentage point), not identical numbers. On GPUs
without native bfloat16 (e.g. Colab's T4), feature extraction uses float16 instead; this was not tested.

## 6. Troubleshooting (problems hit in the original run)

| Symptom | Cause | Fix already in the code |
|---|---|---|
| `CUDA out of memory` while extracting DINO features | ViT-B/8 at 224 px has 785 tokens per image | DINO batch size is 48 (AlexNet 256). Lower it in the last loop of `extract_features.py` if your GPU has < 8 GB |
| `Too many open files` | DataLoader workers pass tensors through file descriptors | `set_sharing_strategy("file_system")` and `.copy()` of the label arrays |
| `torch.hub` HTTP 403 / rate limit | GitHub API limit when downloading DINO | wait and retry, or set a `GITHUB_TOKEN` environment variable |

## 7. Limitations

One dataset (CIFAR-10, upsampled from 32 to 224 px), one supervisor–student pair, linear probes only, one
main seed (plus 5 bootstrap members), no hyper-parameter search. For 10 classes the paper's balanced
adaptive threshold in the confidence loss is replaced by an argmax. PGR needs ground truth, so it is a
research metric, not something a practitioner could compute.

## 8. References

- Burns, C., Izmailov, P., Kirchner, J. H., Baker, B., Gao, L., Aschenbrenner, L., Chen, Y., Ecoffet, A.,
  Joglekar, M., Leike, J., Sutskever, I., & Wu, J. (2024). Weak-to-strong generalization: Eliciting strong
  capabilities with weak supervision. *ICML 2024*, PMLR 235, 4971–5012. arXiv:2312.09390.
- Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging
  properties in self-supervised vision transformers (DINO). *ICCV 2021*, 9630–9640.
- Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images* (CIFAR-10). Tech. report.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2017). ImageNet classification with deep convolutional
  neural networks (AlexNet). *Communications of the ACM*, 60(6), 84–90.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML 2017*, PMLR 70, 1321–1330.
- Sadinle, M., Lei, J., & Wasserman, L. (2019). Least ambiguous set-valued classifiers with bounded error
  levels. *JASA*, 114(525), 223–234.
- Einbinder, B.-S., Feldman, S., Bates, S., Angelopoulos, A. N., Gendler, A., & Romano, Y. (2024).
  Label noise robustness of conformal prediction. *JMLR*, 25(328), 1–66.
