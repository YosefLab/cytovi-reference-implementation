import logging
import warnings
from collections.abc import Sequence
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import rich
import torch
from anndata import AnnData
from scvi import settings
from scvi._types import Number
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
)
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _get_batch_code_from_category
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, UnsupervisedTrainingMixin, VAEMixin
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._module import CytoVAE

logger = logging.getLogger(__name__)


class CytoVI(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
):
    """Adaptation of single-cell Variational Inference :cite:p:`Lopez18` for flow/mass cytometry data.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    protein_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
        * ``'normal'`` - Normal distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.FlowVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.FlowVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    Notes
    -----
    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    _module_cls = CytoVAE
    # _training_plan_cls = AdversarialTrainingPlan
    _data_splitter_cls = DataSplitter
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        protein_likelihood: Literal["normal", "beta"] = "normal",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_backbone_only: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch


        self._model_summary_string = (
            "CytoVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, protein_likelihood: {}, latent_distribution: {}, n_proteins: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            protein_likelihood,
            latent_distribution,
            self.summary_stats.n_vars,
        )

        if REGISTRY_KEYS.PROTEIN_NAN_MASK in self.adata_manager.data_registry:
            nan_layer = self.adata_manager.get_from_registry("nan_layer")
            all_markers = adata.var_names
            backbone_markers = list(all_markers[~np.any(nan_layer == 0, axis=0)])
            self.backbone_markers = backbone_markers
            self.nan_imputation = True
            self.backbone_marker_mask = all_markers.isin(backbone_markers)
            backbone_str = ", ".join(backbone_markers)
            self._model_summary_string += (f", Impute missing markers: {self.nan_imputation}, \nBackbone markers: {backbone_str}")
        else:
            self.backbone_markers = None
            self.backbone_marker_mask = None
            self.nan_imputation = False
            self._model_summary_string += (f", Impute missing markers: {self.nan_imputation}")

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            encode_backbone_only=encode_backbone_only,
            backbone_marker_mask=self.backbone_marker_mask,
            **model_kwargs,
        )


        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        categorical_covariate_keys: Optional[list[str]] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        nan_layer: Optional[str] = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]

        if nan_layer is None and "_nan_mask" in adata.layers:
            msg = "Found nan_layer in adata. Will register nan_layer for missing marker imputation."
            warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
            nan_layer = "_nan_mask"

        if nan_layer is not None:
            anndata_fields.append(LayerField(REGISTRY_KEYS.PROTEIN_NAN_MASK, nan_layer))

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def __repr__(
        self,
    ):
        summary_string = self._model_summary_string
        summary_string += "\nTraining status: {}".format("Trained" if self.is_trained_ else "Not Trained")
        rich.print(summary_string)
        return ""

    # def train(
    #     self,
    #     max_epochs: Optional[int] = None,
    #     lr: float = 4e-3,
    #     use_gpu: Optional[Union[str, int, bool]] = None,
    #     # accelerator: str = "auto",
    #     # devices: Union[int, List[int], str] = "auto",
    #     train_size: float = 0.9,
    #     validation_size: Optional[float] = None,
    #     # shuffle_set_split: bool = True,
    #     batch_size: int = 128,
    #     early_stopping: bool = False,
    #     # check_val_every_n_epoch: Optional[int] = None,
    #     # reduce_lr_on_plateau: bool = True,
    #     # n_steps_kl_warmup: Union[int, None] = None,
    #     # n_epochs_kl_warmup: Union[int, None] = None,
    #     adversarial_classifier: Optional[bool] = None,
    #     plan_kwargs: Optional[dict] = None,
    #     **kwargs,
    # ):
    #     """Trains the model using amortized variational inference.

    #     Parameters
    #     ----------
    #     max_epochs
    #         Number of passes through the dataset.
    #     lr
    #         Learning rate for optimization.
    #     %(param_use_gpu)s
    #     %(param_accelerator)s
    #     %(param_devices)s
    #     train_size
    #         Size of training set in the range [0.0, 1.0].
    #     validation_size
    #         Size of the test set. If `None`, defaults to 1 - `train_size`. If
    #         `train_size + validation_size < 1`, the remaining cells belong to a test set.
    #     shuffle_set_split
    #         Whether to shuffle indices before splitting. If `False`, the val, train, and test set are split in the
    #         sequential order of the data according to `validation_size` and `train_size` percentages.
    #     batch_size
    #         Minibatch size to use during training.
    #     early_stopping
    #         Whether to perform early stopping with respect to the validation set.
    #     check_val_every_n_epoch
    #         Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
    #         or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
    #         every epoch.
    #     reduce_lr_on_plateau
    #         Reduce learning rate on plateau of validation metric (default is ELBO).
    #     n_steps_kl_warmup
    #         Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
    #         Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
    #         to `floor(0.75 * adata.n_obs)`.
    #     n_epochs_kl_warmup
    #         Number of epochs to scale weight on KL divergences from 0 to 1.
    #         Overrides `n_steps_kl_warmup` when both are not `None`.
    #     adversarial_classifier
    #         Whether to use adversarial classifier in the latent space. This helps mixing when
    #         there are missing proteins in any of the batches. Defaults to `True` is missing proteins
    #         are detected.
    #     plan_kwargs
    #         Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
    #         `train()` will overwrite values present in `plan_kwargs`, when appropriate.
    #     **kwargs
    #         Other keyword args for :class:`~scvi.train.Trainer`.
    #     """
    #     if adversarial_classifier is None:
    #         adversarial_classifier = self._use_adversarial_classifier
    #     # n_steps_kl_warmup = (
    #     #     n_steps_kl_warmup
    #     #     if n_steps_kl_warmup is not None
    #     #     else int(0.75 * self.adata.n_obs)
    #     # )
    #     # if reduce_lr_on_plateau:
    #     #     check_val_every_n_epoch = 1

    #     update_dict = {
    #         "lr": lr,
    #         "adversarial_classifier": adversarial_classifier,
    #         # "reduce_lr_on_plateau": reduce_lr_on_plateau,
    #         # "n_epochs_kl_warmup": n_epochs_kl_warmup,
    #         # "n_steps_kl_warmup": n_steps_kl_warmup,
    #     }
    #     if plan_kwargs is not None:
    #         plan_kwargs.update(update_dict)
    #     else:
    #         plan_kwargs = update_dict

    #     # if max_epochs is None:
    #     #     max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

    #     plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

    #     data_splitter = self._data_splitter_cls(
    #         self.adata_manager,
    #         train_size=train_size,
    #         validation_size=validation_size,
    #         # shuffle_set_split=shuffle_set_split,
    #         batch_size=batch_size,
    #     )
    #     training_plan = self._training_plan_cls(self.module, **plan_kwargs)
    #     runner = self._train_runner_cls(
    #         self,
    #         training_plan=training_plan,
    #         data_splitter=data_splitter,
    #         max_epochs=max_epochs,
    #         use_gpu=use_gpu,
    #         # accelerator=accelerator,
    #         # devices=devices,
    #         early_stopping=early_stopping,
    #         # check_val_every_n_epoch=check_val_every_n_epoch,
    #         **kwargs,
    #     )
    #     return runner()

    @torch.inference_mode()
    def posterior_predictive_sample(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: int = 1,
        protein_list: Optional[Sequence[str]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of samples for each cell.
        gene_list
            Names of genes of interest.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        if self.module.protein_likelihood not in ["beta", "normal"]:
            raise ValueError("Invalid protein_likelihood.")

        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if indices is None:
            indices = np.arange(adata.n_obs)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.var_names
            protein_mask = [True if protein in protein_list else False for protein in all_proteins]

        x_new = []
        for tensors in scdl:
            samples = self.module.sample(
                tensors,
                n_samples=n_samples,
            )
            if protein_list is not None:
                samples = samples[:, protein_mask, ...]
            x_new.append(samples)

        x_new = torch.cat(x_new)  # Shape (n_cells, n_genes, n_samples)

        return x_new.numpy()

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = "all",
        protein_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: int = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the normalized (decoded) protein expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - 'all', then the mean across batches is used
            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
            This behaviour affects only proteins that are detected across multiple batches.
            Unobserved proteins are decoded in the batch(es), in which they were measured.
        protein_list
            Return frequencies of expression for a subset of protein.
            This can save memory when working with large datasets and few proteins are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        all_batches = list(np.unique(self.adata_manager.get_from_registry("batch")))

        if self.nan_imputation is True:
            msg = "detected missing proteins between batches - will impute missing markers"
            warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
            backbone_marker_mask=self.backbone_marker_mask
            nan_imputation = True

        else:
            nan_imputation = False

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            protein_mask = [True if protein in protein_list else False for protein in adata.var_names]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                msg = "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
            return_numpy = True

        if transform_batch == "all":
            transform_batch = all_batches
        else:
            transform_batch = _get_batch_code_from_category(
                self.get_anndata_manager(adata, required=True), transform_batch
            )

        if nan_imputation is True:
            decode_batches = all_batches
        else:
            decode_batches = transform_batch

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in decode_batches:
                generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
                inference_kwargs = {"n_samples": n_samples}
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )

                output = generative_outputs["px"].mean
                output = output.cpu().numpy()

                # masking if markers where not measured in respective batch
                if nan_imputation is True:
                    batch_str = "_batch_" + str(batch)
                    batch_marker_mask = adata.var[batch_str]
                    output[..., ~batch_marker_mask] = None

                    # masking if backbone markers of respective batch are not used
                    if transform_batch == [None]:
                        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
                        index_measure_mask = np.array(batch_index != batch).flatten()
                        index_marker_mask = np.outer(index_measure_mask, backbone_marker_mask)

                        output[..., index_marker_mask] = None

                    else:
                        if batch not in transform_batch:
                            output[..., backbone_marker_mask] = None

                output = output[..., protein_mask]

                per_batch_exprs.append(output)

            per_batch_exprs = np.stack(per_batch_exprs)

            exprs += [np.nanmean(per_batch_exprs, axis=0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = np.nanmean(exprs, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[protein_mask],
                index=adata.obs_names[indices],
            )
        else:
            return exprs
