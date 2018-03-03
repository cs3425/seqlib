#!/usr/bin/env python
"""
Functions for a Seqlib obj
"""

import numpy as np
import pandas as pd


class Seqlib:
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites
        self.arr = self.simulate()

    def mutate(self, base):
        diff = set("ACTG") - set(base)
        return np.random.choice(list(diff))

    def simulate(self):
        oseq = np.random.choice(list("ACGT"), size=self.nsites)
        self.arr = np.array([oseq for i in range(self.ninds)])
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        for col in range(self.nsites):
            newbase = self.mutate(self.arr[0, col])
            mask = muts[:, col].astype(bool)
            self.arr[:, col][mask] = newbase
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        self.arr[missing.astype(bool)] = "N"
        return self.arr

    def filter_missing(self, arr, maxfreq):
        freqmissing = np.sum(self.arr == "N", axis=0) / self.arr.shape[0]
        return self.arr[:, freqmissing <= maxfreq]

    def filter_maf(self, arr, minfreq):
        freqs = np.sum(self.arr != self.arr[0], axis=0) / self.arr.shape[0]
        maf = freqs.copy()
        maf[maf > 0.5] = 1 - maf[maf > 0.5]
        maf = self.arr[:, maf > minfreq]
        return self.maf

    def calculate_statistics(self, arr):
        nd = np.var(arr == arr[0], axis=0).mean()
        mf = np.mean(np.sum(arr != arr[0], axis=0) / self.arr.shape[0])
        inv = np.any(arr != arr[0], axis=0).sum()
        var = self.arr.shape[1] - inv
        return pd.Series(
            {"mean nucleotide diversity": nd,
             "mean minor allele frequency": mf,
             "invariant sites": inv,
             "variable sites": var,
             })
