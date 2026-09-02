from typing import Tuple

import numpy as np
import awkward as ak
import torch


FEATURE_NAMES = [
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
]

# One-hot label branches, in class-index order (QCD=0, Hbb=1, ...).
LABEL_NAMES = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
]


def _pad(a, maxlen, value=0, dtype="float32"):
    """
    Pad jagged array to fixed length.

    Source: Particle Transformer JetClass notebook
    https://github.com/jet-universe/particle_transformer/blob/main/notebooks/JetClass101.ipynb
    """
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, : len(trunc)] = trunc
        return x


def _clip(a, a_min, a_max):
    """
    Clip values in array or jagged array.

    Source: Particle Transformer JetClass notebook
    https://github.com/jet-universe/particle_transformer/blob/main/notebooks/JetClass101.ipynb
    """
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def read_branches(tree, entry_stop=None):
    """Read the raw branches used downstream into an awkward record array."""
    return tree.arrays(
        filter_name=["part_*", "jet_pt", "jet_energy", "label_*"],
        entry_stop=entry_stop,
    )


def build_features_and_labels(a, transform_features=True, max_particles=128):
    """
    Build features and labels from raw branches (see read_branches).

    Adapted from the Particle Transformer (ParT), "Particle Transformer for Jet
    Tagging" (https://arxiv.org/abs/2202.03772); standardization constants and
    derived features follow the original implementation:
    https://github.com/jet-universe/particle_transformer

    Returns dict with:
        - pf_features: (batch, 17, max_particles) - particle features
        - pf_mask: (batch, max_particles) - padding mask (1=real, 0=padding)
        - label: (batch,) - class index
    """
    # Fresh wrapper over the same buffers, so field assignments below don't
    # leak derived features into the caller's array
    a = ak.Array(a.layout)

    # Compute derived features
    a["part_mask"] = ak.ones_like(a["part_energy"])
    a["part_pt"] = np.hypot(a["part_px"], a["part_py"])
    a["part_pt_log"] = np.log(a["part_pt"])
    a["part_e_log"] = np.log(a["part_energy"])
    a["part_logptrel"] = np.log(a["part_pt"] / a["jet_pt"])
    a["part_logerel"] = np.log(a["part_energy"] / a["jet_energy"])
    a["part_deltaR"] = np.hypot(a["part_deta"], a["part_dphi"])
    a["part_d0"] = np.tanh(a["part_d0val"])
    a["part_dz"] = np.tanh(a["part_dzval"])

    # Apply standardization (from ParT paper)
    if transform_features:
        a["part_pt_log"] = (a["part_pt_log"] - 1.7) * 0.7
        a["part_e_log"] = (a["part_e_log"] - 2.0) * 0.7
        a["part_logptrel"] = (a["part_logptrel"] - (-4.7)) * 0.7
        a["part_logerel"] = (a["part_logerel"] - (-4.7)) * 0.7
        a["part_deltaR"] = (a["part_deltaR"] - 0.2) * 4.0
        a["part_d0err"] = _clip(a["part_d0err"], 0, 1)
        a["part_dzerr"] = _clip(a["part_dzerr"], 0, 1)

    # Stack and pad features
    out = {}
    out["pf_features"] = np.stack(
        [_pad(a[n], maxlen=max_particles).to_numpy() for n in FEATURE_NAMES], axis=1
    )  # (batch, 17, max_particles)

    out["pf_mask"] = _pad(a["part_mask"], maxlen=max_particles).to_numpy()
    # (batch, max_particles) - 1 for real particles, 0 for padding

    # Convert one-hot labels to class indices
    label_array = np.stack(
        [a[n].to_numpy().astype("int") for n in LABEL_NAMES], axis=1
    )  # (batch, 10)
    out["label"] = np.argmax(label_array, axis=1)  # (batch,) - class index

    return out


def vectorize(
    tree, max_particles: int = 128, entry_stop=None, perturb=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorize an opened ROOT tree into fixed-size CPU tensors.

    Args:
        tree: An opened uproot tree (e.g. file["tree"]).
        max_particles: Particles per jet after padding/clipping.
        entry_stop: Optional number of jets to read from the tree.
        perturb: Optional callable applied to the raw awkward record before any
            derived features are computed, so the perturbation propagates
            consistently through every derived feature.

    Returns:
        x: (num_jets, 17, max_particles) - particle features
        mask: (num_jets, max_particles) - padding mask (True=padding, False=real)
        labels: (num_jets,) - class indices
    """
    a = read_branches(tree, entry_stop=entry_stop)
    if perturb is not None:
        a = perturb(a)
    data = build_features_and_labels(a, max_particles=max_particles)

    x = torch.from_numpy(data["pf_features"]).float()
    mask = torch.from_numpy(data["pf_mask"]).float() == 0  # invert: True for padding
    labels = torch.from_numpy(data["label"]).long()

    return x, mask, labels
