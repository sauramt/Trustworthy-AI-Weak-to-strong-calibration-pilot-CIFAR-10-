# What every number in `results/metrics.json` means

All metrics are computed on the 10,000 CIFAR-10 test images with their **true** labels, unless the key
says otherwise. "Confidence" = the largest softmax probability of a prediction.

## Top level

| Key | Meaning |
|---|---|
| `setting` | the configuration of the run: models, split sizes (20k / 25k / 5k / 10k), epochs, number of bootstrap members, α of the confidence loss, conformal level |
| `weak`, `naive`, `conf`, `ceiling` | one block per model (see the README table for what each model is) |
| `runtime_s` | seconds for `w2s_pilot.py` (probes only, features already extracted) |

## Inside each model block

### Accuracy and the paper's metric

| Key | Meaning |
|---|---|
| `acc` | top-1 accuracy |
| `pgr` | performance gap recovered = (acc_student − acc_weak) / (acc_ceiling − acc_weak). 0 = no better than the supervisor, 1 = as good as training on the truth. Only for `naive` and `conf` |
| `pgr_best_epoch` | the same with the student's best epoch on the test set. **Uses ground truth to pick the epoch, so it is an upper bound**, only reported to show overfitting to weak labels |
| `member_acc` | accuracy of each of the 5 bootstrap members |
| `weak_label_acc_on_w2s_split` | (weak only) how often the weak labels given to the student are right: 79.6 % |

### Calibration (are the probabilities honest?)

| Key | Meaning |
|---|---|
| `ece` | expected calibration error, 15 equal-width confidence bins: the average gap between confidence and accuracy, weighted by bin size (Guo et al., 2017). 0 = perfectly calibrated |
| `nll` | negative log-likelihood of the true class (lower is better; punishes confident mistakes hard) |
| `brier` | Brier score: squared distance between the probability vector and the one-hot truth (lower is better) |
| `mean_conf` | average confidence. Compare with `acc`: conf > acc means over-confident, conf < acc means under-confident |
| `ensemble` | the same five metrics for the average of the 5 bootstrap members |

### Does the student copy its teacher's mistakes, and does it know?

| Key | Meaning |
|---|---|
| `n_weak_errors` | test images the weak supervisor gets wrong (2,025) |
| `n_corrected` | of those, how many the student gets right |
| `n_imitated` | student errors where the student predicts **exactly the supervisor's wrong class** ("copied errors") |
| `share_errors_imitated` | n_imitated / all student errors |
| `mean_conf_correct` | student's mean confidence on its correct predictions |
| `mean_conf_imitated_errors` | ... on its copied errors |
| `mean_conf_other_errors` | ... on its own, non-copied errors |
| `auroc_imitated_vs_correct_maxprob` | how well low confidence separates copied errors from correct answers (0.5 = chance, 1 = perfect) |

### Can the student's own uncertainty flag its errors? (label-free error detection)

AUROC of a score for "this prediction is wrong" (higher score = more suspicious):

| Key | Score |
|---|---|
| `auroc_err_maxprob` | 1 − confidence |
| `auroc_err_entropy` | predictive entropy of the softmax |
| `auroc_err_ens_mi` | mutual information of the 5-member ensemble (the epistemic part of the entropy: total entropy − mean member entropy), scored against the ensemble's own errors |

In this pilot, 1 − confidence and entropy are the best detectors (nearly tied: 0.88 / 0.86 naive,
0.91 / 0.90 confidence loss, 0.948 / 0.948 ceiling). The ensemble's mutual information is lower for every
model (0.82, 0.88, 0.94). With only 5 bootstrap members of a linear probe this is not a verdict on ensembles.

### Conformal prediction (prediction sets with a coverage target)

Split conformal with the LAC score s = 1 − p(label) (Sadinle et al., 2019), target coverage 90 %.
The 5k calibration images are scored with either label type:

| Key | Meaning |
|---|---|
| `conformal_weak_cal` | calibration with the supervisor's labels (argmax of its probabilities): what a practitioner without ground truth can do |
| `conformal_true_cal` | calibration with true labels: the oracle reference |
| `.qhat` | the threshold: a class enters the set when 1 − p(class) ≤ qhat |
| `.coverage_true` | share of test images whose **true** label is in the set (target 0.90) |
| `.mean_set_size` | average number of classes per set. Can be below 1: LAC allows empty sets when no class is probable enough |

Why weak-label calibration over-covers: where the student is right and the weak label is wrong, the
score 1 − p(weak label) is close to 1, so the 90 % quantile (qhat) is pushed up and every set grows.
Einbinder et al. (2024) show that this kind of label noise makes conformal prediction conservative.

## Other files

| File | Content |
|---|---|
| `curves.csv` | `method, fit, epoch, test_acc`: ground-truth test accuracy after every epoch of the main fit. Logged for analysis only; it never influences training |
| `test_probs.npz` | arrays `y` (true labels), `weak`, `naive`, `conf`, `ceiling` (10000×10 probabilities of the main fit), and `naive_ens`, `conf_ens`, `ceiling_ens` (ensemble means). Load with `np.load("results/test_probs.npz")` |
| `pilot_table.csv` | one row per model with the headline columns |
| `features_meta.json` | feature shapes, torch version, device of the original run |
