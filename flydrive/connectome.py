"""Loader for the Janelia/Google MaleCNS v1.0 connectome.

Builds the whole central nervous system -- brain, optic lobes, neck connective
and ventral nerve cord -- as a signed sparse adjacency matrix that can be run as
a recurrent network.

Signs come from the released neurotransmitter predictions rather than being
learned, which is the one place a connectome really does pin down the dynamics:
acetylcholine excites, GABA and glutamate inhibit (Drosophila glutamate acts on
GluCl channels), and histamine inhibits -- which is how the photoreceptors, who
are histaminergic, hyperpolarise L1/L2 in the light and create the ON and OFF
channels in the first place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Which neurons count as reconstructed well enough to simulate.
KEEP_STATUS = ("Roughly traced", "Reviewed", "Prelim Roughly traced")

NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": 0.0,
    "octopamine": 0.0,
    "serotonin": 0.0,
    "unclear": 0.0,
}


@dataclass
class Connectome:
    body: np.ndarray          # (N,) body ids, sorted
    type: np.ndarray          # (N,) cell type strings
    superclass: np.ndarray    # (N,)
    side: np.ndarray          # (N,) 'L' / 'R' / 'M'
    hex1: np.ndarray          # (N,) optic-lobe column, NaN where unassigned
    hex2: np.ndarray
    sign: np.ndarray          # (N,) presynaptic sign from the predicted transmitter
    pre: np.ndarray           # (E,) int32 row index
    post: np.ndarray          # (E,) int32 column index
    weight: np.ndarray        # (E,) float32 synapse counts, already signed

    @property
    def n(self):
        return self.body.shape[0]

    @property
    def n_edges(self):
        return self.pre.shape[0]

    def mask_type(self, *names):
        return np.isin(self.type, names)

    def mask_superclass(self, *names):
        return np.isin(self.superclass, names)

    def summary(self):
        exc = float((self.weight > 0).mean())
        return (f"MaleCNS v1.0: {self.n:,} neurons, {self.n_edges:,} edges, "
                f"{int(np.abs(self.weight).sum()):,} synapses, "
                f"{exc:.1%} excitatory")


def load(data_dir: str = DATA, verbose: bool = True) -> Connectome:
    ann = pd.read_feather(os.path.join(data_dir, "body-annotations.feather"))
    ann = ann[ann["statusLabel"].isin(KEEP_STATUS)].copy()
    ann = ann.sort_values("bodyId").reset_index(drop=True)
    body = ann["bodyId"].to_numpy(np.int64)
    if verbose:
        print(f"  neurons kept: {len(body):,}")

    # -- presynaptic sign from the predicted transmitter -------------------
    nt = pd.read_feather(os.path.join(data_dir, "body-neurotransmitters.feather"),
                         columns=["body", "consensus_nt"])
    nt = nt.drop_duplicates("body")
    nt_map = dict(zip(nt["body"].to_numpy(np.int64), nt["consensus_nt"].to_numpy()))
    sign = np.array([NT_SIGN.get(str(nt_map.get(b, "unclear")).lower(), 0.0) for b in body],
                    dtype=np.float32)
    if verbose:
        vals, cnt = np.unique(sign, return_counts=True)
        print("  presynaptic signs:", dict(zip(vals.tolist(), cnt.tolist())))

    # -- edges -------------------------------------------------------------
    w = pd.read_feather(os.path.join(data_dir, "connectome-weights.feather"),
                        columns=["body_pre", "body_post", "weight"])
    pre_b = w["body_pre"].to_numpy(np.int64)
    post_b = w["body_post"].to_numpy(np.int64)
    wt = w["weight"].to_numpy(np.float32)
    del w

    # Map body ids to dense indices with a sorted search, which avoids building
    # a 200k-entry python dict lookup over 25M edges.
    pi = np.searchsorted(body, pre_b)
    qi = np.searchsorted(body, post_b)
    pi_ok = (pi < len(body)) & (body[np.clip(pi, 0, len(body) - 1)] == pre_b)
    qi_ok = (qi < len(body)) & (body[np.clip(qi, 0, len(body) - 1)] == post_b)
    keep = pi_ok & qi_ok
    pre, post, wt = pi[keep].astype(np.int32), qi[keep].astype(np.int32), wt[keep]
    if verbose:
        print(f"  edges kept: {len(pre):,} of {len(keep):,}")

    signed = wt * sign[pre]
    nz = signed != 0.0
    pre, post, signed = pre[nz], post[nz], signed[nz]
    if verbose:
        print(f"  edges with a known sign: {len(pre):,}")

    return Connectome(
        body=body,
        type=ann["type"].astype(str).to_numpy(),
        superclass=ann["superclass"].astype(str).to_numpy(),
        side=ann["somaSide"].astype(str).to_numpy(),
        hex1=ann["assignedOlHex1"].to_numpy(np.float32),
        hex2=ann["assignedOlHex2"].to_numpy(np.float32),
        sign=sign, pre=pre, post=post, weight=signed,
    )


def photoreceptor_columns(c: Connectome) -> np.ndarray:
    """Give every R1-R6 photoreceptor the retinotopic column of the L1 it drives.

    The release assigns hexagonal coordinates to lamina and medulla cells but
    not to the photoreceptors themselves, so the column is inherited across the
    strongest R -> L1 connection.  Returns ``(N, 2)`` of (hex1, hex2), NaN where
    unknown.
    """
    out = np.stack([c.hex1, c.hex2], axis=1).copy()
    r_idx = np.where(c.type == "R1-R6")[0]
    l1_idx = np.where(c.type == "L1")[0]
    if len(r_idx) == 0 or len(l1_idx) == 0:
        return out

    is_r = np.zeros(c.n, bool); is_r[r_idx] = True
    is_l1 = np.zeros(c.n, bool); is_l1[l1_idx] = True
    sel = is_r[c.pre] & is_l1[c.post]
    pre, post, w = c.pre[sel], c.post[sel], np.abs(c.weight[sel])

    # strongest R -> L1 edge per photoreceptor
    order = np.lexsort((-w, pre))
    pre, post = pre[order], post[order]
    first = np.ones(len(pre), bool)
    first[1:] = pre[1:] != pre[:-1]
    out[pre[first]] = np.stack([c.hex1[post[first]], c.hex2[post[first]]], axis=1)
    return out
