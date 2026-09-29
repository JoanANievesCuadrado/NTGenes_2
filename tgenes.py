"""T-genes analysis: genes deregulated in tumor samples."""

import numpy as np
import pandas as pd
from typing import Tuple


def get_tgenes(data: pd.DataFrame, normal_mask: np.ndarray,
               threshold: float = 0.1, padding: float = 0.1) -> Tuple:
    """Identify T-genes (tumor-deregulated genes).

    T-genes are genes where tumor expression falls outside normal range
    (either above normal_max or below normal_min).

    Parameters
    ----------
    data : pd.DataFrame
        Gene expression matrix (samples x genes)
    normal_mask : np.ndarray
        Boolean mask indicating normal samples
    threshold : float
        Proportion threshold (default 0.1 for 10% of tumor samples)
    padding : float
        Offset for threshold comparison (default 0.1)

    Returns
    -------
    tuple
        (tdata, tgenes, genes_above, genes_bellow, max_above, min_bellow,
         genes_above_only, genes_both, genes_bellow_only)
    """
    normal_data = data[normal_mask]
    tumor_data = data[~normal_mask]

    # Padded normal range: these are the thresholds used both to select the
    # T-genes and later to binarize the data, so they must be the same values.
    normal_min = normal_data.min()
    normal_max = normal_data.max()
    n_tumor = tumor_data.shape[0]

    f_bellow = (tumor_data < normal_min - padding).sum() / n_tumor
    f_above = (tumor_data > normal_max + padding).sum() / n_tumor

    ibellow, = np.where(f_bellow > threshold)
    iabove, = np.where(f_above > threshold)

    genes_above_set = set(data.columns[iabove])
    genes_bellow_set = set(data.columns[ibellow])

    # Classify genes according to Mathematica logic
    genes_both = genes_above_set & genes_bellow_set  # "to"
    genes_above_only = genes_above_set - genes_both  # "ta"
    genes_bellow_only = genes_bellow_set - genes_both  # "tb"

    genes_both = np.array(list(genes_both))
    genes_above_only = np.array(list(genes_above_only))
    genes_bellow_only = np.array(list(genes_bellow_only))

    genes_above, max_above = data.columns[iabove], normal_max.iloc[iabove]
    genes_bellow, min_bellow = data.columns[ibellow], normal_min.iloc[ibellow]
    tgenes = np.array(list(genes_above_set | genes_bellow_set))
    tdata = tumor_data[tgenes]

    return tdata, tgenes, genes_above, genes_bellow, max_above, min_bellow, genes_above_only, genes_both, genes_bellow_only
