"""N-genes analysis: genes deregulated in normal samples."""

import numpy as np
import pandas as pd
from typing import Tuple


def get_ngenes(data: pd.DataFrame, normal_mask: np.ndarray,
               threshold: float = 0.05, padding: float = 0.1) -> Tuple:
    """Identify N-genes (normal-deregulated genes).

    N-genes are genes where normal expression falls outside tumor range
    (either above tumor_max or below tumor_min).

    Parameters
    ----------
    data : pd.DataFrame
        Gene expression matrix (samples x genes)
    normal_mask : np.ndarray
        Boolean mask indicating normal samples
    threshold : float
        Proportion threshold (default 0.05 for 5% of normal samples)
    padding : float
        Offset for threshold comparison (default 0.1)

    Returns
    -------
    tuple
        (ndata, ngenes, genes_above, genes_bellow, max_above, min_bellow,
         genes_above_only, genes_both, genes_bellow_only)
    """
    normal_data = data[normal_mask]
    tumor_data = data[~normal_mask]

    # Padded tumor range: these are the thresholds used both to select the
    # N-genes and later to binarize the data, so they must be the same values.
    tumor_min = tumor_data.min() - padding
    tumor_max = tumor_data.max() + padding
    n_normal = normal_data.shape[0]

    f_bellow = (normal_data < tumor_min).sum() / n_normal
    f_above = (normal_data > tumor_max).sum() / n_normal

    ibellow, = np.where(f_bellow > threshold)
    iabove, = np.where(f_above > threshold)

    genes_above_set = set(data.columns[iabove])
    genes_bellow_set = set(data.columns[ibellow])

    # Classify genes according to Mathematica logic
    genes_both = genes_above_set & genes_bellow_set  # "no"
    genes_above_only = genes_above_set - genes_both  # "na"
    genes_bellow_only = genes_bellow_set - genes_both  # "nb"

    genes_both = np.array(list(genes_both))
    genes_above_only = np.array(list(genes_above_only))
    genes_bellow_only = np.array(list(genes_bellow_only))

    genes_above, max_above = data.columns[iabove], tumor_max.iloc[iabove]
    genes_bellow, min_bellow = data.columns[ibellow], tumor_min.iloc[ibellow]
    ngenes = np.array(list(genes_above_set | genes_bellow_set))
    ndata = normal_data[ngenes]

    return ndata, ngenes, genes_above, genes_bellow, max_above, min_bellow, genes_above_only, genes_both, genes_bellow_only
