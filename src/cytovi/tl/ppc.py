from typing import Literal, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import pearsonr, spearmanr
from scvi.model.base import BaseModelClass
from scvi_criticism import PPC
from scvi_criticism._constants import (
    DATA_VAR_RAW,
    METRIC_CV_CELL,
    METRIC_CV_GENE,
)
from scvi_criticism._ppc import _make_dataset_dense
from sklearn.metrics import mean_absolute_error as mae

Dims = Literal["cells", "features"]
METRIC_MEAN_CELL = "mean_cell"
METRIC_MEAN_GENE = "mean_gene"
METRIC_STD_CELL = "std_cell"
METRIC_STD_GENE = "std_gene"


class PPC(PPC):
    """
    Posterior predictive checks for CytoVI models.

    Inherits from scvi-criticism's PPC class but adds additional funcitonality for non-count flow cytometry data.
    """

    def __init__(
        self,
        adata: AnnData,
        models_dict: dict[str, BaseModelClass],
        layer_key: Optional[str] = None,
        n_samples: int = 10,
    ):
        count_layer_key = layer_key
        super().__init__(adata, models_dict, count_layer_key, n_samples)

    def coefficient_of_variation(self, dim: Dims = "cells") -> None:
        """

        Note: we needed to remove the sqrt trick for faster computation of the std because flow data can be negative

        Calculate the coefficient of variation (CV) for each model and the raw counts.

        The CV is computed over the cells or features dimension per sample. The mean CV is then
        computed over all samples.

        Parameters
        ----------
        dim
            Dimension to compute CV over.
        """
        identifier = METRIC_CV_CELL if dim == "features" else METRIC_CV_GENE
        mean = self.samples_dataset.mean(dim=dim, skipna=False)
        std = self.samples_dataset.std(dim=dim, skipna=False)
        # now compute the CV
        cv = std / mean
        cv = _make_dataset_dense(cv)
        cv_mean = cv.mean(dim="samples", skipna=True)
        cv_mean[DATA_VAR_RAW].data = np.nan_to_num(cv_mean[DATA_VAR_RAW].data)
        self.metrics[identifier] = cv_mean.to_dataframe()

    def mean(self, dim: Dims = "cells") -> None:
        """

        Note: we needed to remove the sqrt trick for faster computation of the std because flow data can be negative

        Calculate the mean for each model and the raw counts.

        The mean is computed over the cells or features dimension per sample. The mean of the means is then
        computed over all samples.

        Parameters
        ----------
        dim
            Dimension to compute mean over.
        """
        identifier = METRIC_MEAN_CELL if dim == "features" else METRIC_MEAN_GENE
        mean = self.samples_dataset.mean(dim=dim, skipna=False)
        mean = _make_dataset_dense(mean)
        mean_mean = mean.mean(dim="samples", skipna=True)
        mean_mean[DATA_VAR_RAW].data = np.nan_to_num(mean_mean[DATA_VAR_RAW].data)
        self.metrics[identifier] = mean_mean.to_dataframe()

    def std(self, dim: Dims = "cells") -> None:
        """
        Calculate the std for each model and the raw counts.

        The std is computed over the cells or features dimension per sample. The mean of the std is then
        computed over all samples.

        Parameters
        ----------
        dim
            Dimension to compute mean over.
        """
        identifier = METRIC_STD_CELL if dim == "features" else METRIC_STD_GENE
        std = self.samples_dataset.std(dim=dim, skipna=False)
        std = _make_dataset_dense(std)
        std_mean = std.mean(dim="samples", skipna=True)
        std_mean[DATA_VAR_RAW].data = np.nan_to_num(std_mean[DATA_VAR_RAW].data)
        self.metrics[identifier] = std_mean.to_dataframe()

    def compute_summary_statistics(self, metric="all") -> None:
        """Compute summary statistics for each model and metric."""
        if metric == "all":
            metrics = self.metrics.keys()
        elif isinstance(metric, str):
            metrics = [metric]

        models = self.models.keys()
        summary_stats = {}

        for metric_oi in metrics:
            summary_dict_metric = {}

            for model_oi in models:
                # compute summary stats per model
                mae_metric = mae(self.metrics[metric_oi][model_oi], self.metrics[metric_oi][DATA_VAR_RAW])
                pearsonr_metric = pearsonr(self.metrics[metric_oi][model_oi], self.metrics[metric_oi][DATA_VAR_RAW])
                spearmanr_metric = spearmanr(self.metrics[metric_oi][model_oi], self.metrics[metric_oi][DATA_VAR_RAW])
                summary_dict_model = {
                    "mae": mae_metric,
                    "pearsonr": pearsonr_metric[0],
                    "spearmanr": spearmanr_metric[0],
                }

                # store the summary stats for each model
                summary_dict_metric[model_oi] = summary_dict_model

            # store the summary stats for each metric
            summary_stats[metric_oi] = pd.DataFrame(summary_dict_metric)

        self.summary_statistics = summary_stats
