#!/usr/bin/env python
"""
Functions for a Seqlib obj
"""

import numpy as np
import pandas as pd
import copy


class Seqlib:
    def __init__(self, ninds, nsites):
        self.ninds = ninds
        self.nsites = nsites
        self.seqs = self._simulate()
        self.maf = self._get_maf()

    def _mutate(self, base):
        diff = set("ACTG") - set(base)
        return np.random.choice(list(diff))

    def _simulate(self):
        oseq = np.random.choice(list("ACGT"), size=self.nsites)
        self.self = np.array([oseq for i in range(self.ninds)])
        muts = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        for col in range(self.nsites):
            newbase = self._mutate(self.self[0, col])
            mask = muts[:, col].astype(bool)
            self.self[:, col][mask] = newbase
        missing = np.random.binomial(1, 0.1, (self.ninds, self.nsites))
        self.self[missing.astype(bool)] = "N"
        return self.self

    def _get_maf(self):
        "returns the maf of the full seqarray while not counting Ns"
        maf = np.zeros(self.nsites)
        for col in range(self.nsites):
            thiscol = self.self[:, col]
            nmask = thiscol != "N"
            no_n_len = np.sum(nmask)
            first_non_n_base = thiscol[nmask][0]
            freq = np.sum(thiscol[nmask] != first_non_n_base) / no_n_len
            if freq > 0.5:
                maf[col] = 1 - freq
            else:
                maf[col] = freq
        return maf

    def _filter_missing(self, maxmissing):
        freqmissing = np.sum(self.self == "N", axis=0) / self.self.shape[0]
        return freqmissing > maxmissing

    def _filter_maf(self, minmaf):
        return self.maf < minmaf

    def filter(self, minmaf, maxmissing):
        """
        Applies maf and missing filters to the array
        Parameters
        ----------
        minmaf: float
            The minimum minor allele frequency. Filter columns below this.
        maxmissing: float
            The maximum prop. missing data. Filter columns with prop Ns > this.
        """
        filter1 = self._filter_maf(minmaf)
        filter2 = self._filter_missing(maxmissing)
        fullfilter = filter1 + filter2
        return self.self[:, np.invert(fullfilter)]

    def filter_seqlib(self, minmaf, maxmissing):
        """
        Applies maf and missing filters to the array and returns a copy 
        of the seqlib object where the .seqs array has been filtered
        Parameters
        ----------
        minmaf: float
            The minimum minor allele frequency. Filter columns below this.
        maxmissing: float
            The maximum prop. missing data. Filter columns with prop Ns > this.
        """
        # apply filters to get new array size
        newseqs = self.filter(minmaf, maxmissing)

        # make a new copy of the seqlib object
        newself = copy.deepcopy(self)       
        newself.__init__(newseqs.shape[0], newseqs.shape[1]) 

        # store the array (overwrite it)
        newself.seqs = newseqs

        # call the _get_maf to match new array
        newself._get_maf()
        return newself

    def calculate_statistics(self):
        """
        Returns a dataframe of statistics on the seqs array. The earlier 
        example from the notebook had a bug where var and inv were switched.
        """
        if self.seqs.size:
            nd = np.var(self.seqs == self.seqs[0], axis=0).mean()
            mf = np.mean(
                np.sum(self.seqs != self.seqs[0], axis=0) / self.seqs.shape[0])
            inv = np.all(self.seqs == self.seqs[0], axis=0).sum()
            var = self.seqs.shape[1] - inv
            return pd.Series(
                {"mean nucleotide diversity": nd,
                 "mean minor allele frequency": mf,
                 "invariant sites": inv,
                 "variable sites": var,
                })
        else:
            print("seqs array is empty")