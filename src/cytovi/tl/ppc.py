from typing import Literal, Optional

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from scipy import linalg
from scipy.stats import pearsonr, spearmanr
from scvi.criticism._constants import (
    DATA_VAR_RAW,
    METRIC_CV_CELL,
    METRIC_CV_GENE,
)
from scvi.criticism._ppc import PosteriorPredictiveCheck, _make_dataset_dense
from scvi.model.base import BaseModelClass
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import mean_absolute_error as mae

Dims = Literal["cells", "features"]
METRIC_MEAN_CELL = "mean_cell"
METRIC_MEAN_GENE = "mean_gene"
METRIC_STD_CELL = "std_cell"
METRIC_STD_GENE = "std_gene"

FA_VAR = "FA"


class PosteriorPredictiveCheck(PosteriorPredictiveCheck):
    """
    Posterior predictive checks for CytoVI models.

    Inherits from scvi-criticism's PPC class but adds additional funcitonality for non-count flow cytometry data and additional benchmarking methods.
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

    def store_FA_samples(self, n_samples: int = 10, train_indices = None) -> None:
        """
        Store the samples from the FA model in the samples dataset.; FA code from totalVI reproducibility

        Parameters
        ----------
        n_samples
            Number of samples to store.
        """
        if train_indices is None:
            data_train = self.adata.layers[self.count_layer_key]
        else:
            data_train = self.adata.layers[self.count_layer_key][train_indices, :]

        data_trans = self.adata.layers[self.count_layer_key]

        fa = FactorAnalysis()
        fa.fit(data_train)

        # transform gives the posterior mean
        z_mean = fa.transform(data_trans)
        Ih = np.eye(len(fa.components_))

        # W is n_components by n_features, code below from sklearn implementation
        Wpsi = fa.components_ / fa.noise_variance_
        z_cov = linalg.inv(Ih + np.dot(Wpsi, fa.components_.T))

        # sample z's
        z_samples = np.random.multivariate_normal(
                    np.zeros(data_trans.shape[1], dtype=np.float32),
                    cov=z_cov,
                    size=(data_trans.shape[0], n_samples),
        )

        # cells by n_components by posterior samples
        z_samples = np.swapaxes(z_samples, 1, 2)

        # add mean to all samples
        z_samples += z_mean[:, :, np.newaxis]

        x_samples = np.zeros(
                    (data_trans.shape[0], data_trans.shape[1], n_samples),
                    dtype=np.float32,
        )

        for i in range(n_samples):
            x_mean = np.matmul(z_samples[:, :, i], fa.components_)
            x_sample = np.random.normal(x_mean, scale=np.sqrt(fa.noise_variance_))
            # add back feature means
            x_samples[:, :, i] = x_sample + fa.mean_

        reconstruction = x_samples

        reconstruction_array = xr.DataArray(reconstruction, dims = ['cells', 'features', 'samples'])
        self.samples_dataset[FA_VAR] = reconstruction_array
        self.models[FA_VAR] = [FA_VAR]


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

    def compute_metrics(self, metric: Literal["all", "cv", "mean", "std"] = "all") -> None:
        """Compute metrics for each model."""
        if metric == "all":
            metrics = ["cv", "mean", "std"]
        elif isinstance(metric, str):
            metrics = [metric]

        for metric_oi in metrics:
            if metric_oi == "cv":
                self.coefficient_of_variation("cells")
                self.coefficient_of_variation("features")
            elif metric_oi == "mean":
                self.mean("cells")
                self.mean("features")
            elif metric_oi == "std":
                self.std("cells")
                self.std("features")

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
