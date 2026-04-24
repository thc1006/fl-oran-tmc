"""OOD split by training_config (tr) — model must generalise to unseen RBG
allocations.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict:
        return {
            "train_rows": len(self.train),
            "val_rows": len(self.val),
            "test_rows": len(self.test),
            "train_tr": sorted(self.train["tr"].unique().tolist()) if "tr" in self.train.columns else [],
            "val_tr": sorted(self.val["tr"].unique().tolist()) if "tr" in self.val.columns else [],
            "test_tr": sorted(self.test["tr"].unique().tolist()) if "tr" in self.test.columns else [],
        }


def ood_split_by_tr(
    df: pd.DataFrame,
    train_tr: list[int] = tuple(range(22)),
    val_tr: list[int] = (22, 23, 24),
    test_tr: list[int] = (25, 26, 27),
) -> Split:
    """Partition by ``tr`` column.

    Default: tr0-21 train (22 configs), tr22-24 val (3), tr25-27 test (3).
    """
    tr_s = set(train_tr)
    tr_v = set(val_tr)
    tr_t = set(test_tr)
    overlap = tr_s & tr_v, tr_s & tr_t, tr_v & tr_t
    if any(overlap):
        raise ValueError(f"overlapping tr sets: {overlap}")

    tr = df[df["tr"].isin(tr_s)].reset_index(drop=True)
    va = df[df["tr"].isin(tr_v)].reset_index(drop=True)
    te = df[df["tr"].isin(tr_t)].reset_index(drop=True)
    split = Split(train=tr, val=va, test=te)
    log.info("OOD split: %s", split.summary())
    return split
