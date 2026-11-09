"""Extract frozen image features for the weak-to-strong (W2S) calibration pilot.

Purpose : Reproduce the representation side of Burns et al. (2024), App. "Self-supervised vision
          models": a pretrained AlexNet is the WEAK supervisor and a self-supervised DINO ViT-B/8 is
          the STRONG student. Both networks stay frozen; only linear probes are trained later
          (w2s_pilot.py), which the paper itself recommends for fast iteration (App. D.2).
          We use CIFAR-10 instead of ImageNet (ImageNet is not freely downloadable; CIFAR-10 is).
Inputs  : CIFAR-10 (downloaded by torchvision into ./data on first run).
Outputs : ./features/{alexnet,dino}_{train,test}.npy (float16) and ./features/labels_{train,test}.npy
Run     : python extract_features.py      (needs a CUDA GPU in practice: ~15 min on an 8 GB RTX 5060 laptop GPU.
                                          CPU works but takes many hours. Colab: see run_on_colab.ipynb)
Network : first run downloads CIFAR-10 (~170 MB), AlexNet weights (~230 MB) and DINO ViT-B/8 via
          torch.hub from GitHub (~330 MB), into ./data and the torch hub cache (~/.cache/torch).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as T

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"                     # CIFAR-10 download target (not shared: regenerated)
FEAT = HERE / "features"                 # feature cache (not shared: ~0.6 GB, regenerated)
OUT = HERE / "results"                   # small metadata, shared
for d in (DATA, FEAT, OUT):
    d.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# Mixed precision for the forward pass. The original run used bfloat16 (RTX 5060). GPUs without native
# bfloat16 (e.g. Colab's T4) fall back to float16. Features then differ slightly; we expect this
# not to move the metrics noticeably, but only the bfloat16 run was actually done.
# Native bfloat16 needs compute capability >= 8.0 (Ampere or newer). is_bf16_supported() is not used
# because recent torch versions also return True when bfloat16 is only emulated (slow).
AMP_DTYPE = (torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)
# DataLoader worker processes: 4 on the original laptop; capped by the CPUs available (Colab has 2).
N_WORKERS = min(4, os.cpu_count() or 1)
# The first DINO run died with "Too many open files" (DataLoader workers pass tensors through file
# descriptors). The file_system sharing strategy avoids the per-tensor descriptors.
torch.multiprocessing.set_sharing_strategy("file_system")

# Both backbones were pretrained on ImageNet at 224 px with ImageNet normalisation, so CIFAR's
# 32x32 images are upsampled to 224 and normalised with the ImageNet mean/std.
tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def alexnet_backbone():
    """AlexNet (torchvision ImageNet weights) up to fc7 + ReLU: a 4096-d penultimate feature.

    The ImageNet classification head (fc8) is dropped, so the weak supervisor is a linear probe on
    the same penultimate representation that AlexNet's own classifier uses.
    """
    m = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Sequential(*list(m.classifier.children())[:-1])  # drop fc8
    return m


def dino_backbone():
    """DINO ViT-B/8 (Caron et al., 2021), the strong student of the paper's vision experiment.

    Self-supervised: it never saw labels during pretraining, so there is no label leakage
    (the "pretraining leakage" disanalogy the paper discusses does not apply here).
    torch.hub returns a model whose forward() gives the 768-d CLS embedding.
    """
    return torch.hub.load("facebookresearch/dino:main", "dino_vitb8")


@torch.no_grad()
def extract(model, split_train, name, bs):
    """Run the frozen backbone over one CIFAR-10 split and save its features as float16."""
    ds = torchvision.datasets.CIFAR10(DATA, train=split_train, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=N_WORKERS)
    model = model.to(device).eval()
    feats, labels = [], []
    for x, y in dl:
        # autocast (bfloat16, or float16 as fallback) halves memory and time; features are cast back to fp32.
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=device == "cuda"):
            f = model(x.to(device, non_blocking=True))
        feats.append(f.float().cpu().numpy().astype(np.float16))
        labels.append(y.numpy().copy())   # copy: a view would keep the worker's shared-memory file open
    split = "train" if split_train else "test"
    np.save(FEAT / f"{name}_{split}.npy", np.concatenate(feats))
    np.save(FEAT / f"labels_{split}.npy", np.concatenate(labels))
    return np.concatenate(feats).shape


meta = {"device": device, "torch": torch.__version__, "amp_dtype": str(AMP_DTYPE)}
# ViT-B/8 at 224 px has 785 tokens: batch 256 ran out of the 8 GB GPU memory, so DINO uses 48.
for name, ctor, bs in [("alexnet", alexnet_backbone, 256), ("dino", dino_backbone, 48)]:
    t0 = time.time()
    model = ctor()
    for split_train in (True, False):
        split = "train" if split_train else "test"
        if (FEAT / f"{name}_{split}.npy").exists():          # resume: skip splits already extracted
            shape = np.load(FEAT / f"{name}_{split}.npy", mmap_mode="r").shape
            meta[f"{name}_{split}_shape"] = list(shape)
            continue
        shape = extract(model, split_train, name, bs)
        print(f"{name} {'train' if split_train else 'test'} {shape} {time.time() - t0:.0f}s", flush=True)
        meta[f"{name}_{'train' if split_train else 'test'}_shape"] = list(shape)
    del model
    torch.cuda.empty_cache()
(OUT / "features_meta.json").write_text(json.dumps(meta, indent=2))
print("done", flush=True)
