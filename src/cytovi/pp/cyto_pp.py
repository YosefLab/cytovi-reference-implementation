import warnings
from typing import Optional

import anndata as ad
import numpy as np
from anndata import AnnData
from scvi import settings
from sklearn.preprocessing import MinMaxScaler

# write checks for all pp params as well -> move checks into general utils


def arcsinh(
    adata: AnnData,
    raw_layer_key: str = "raw",
    transformed_layer_key: str = "transformed",
    global_scaling_factor: float = 5,
    scaling_dict: Optional[dict[str, float]] = None,
    transform_scatter: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Apply the arcsinh transformation to the 'raw' layer of an AnnData object with variable scaling factors and saves in a new layer.

    Parameters
    ----------
        adata (AnnData): The AnnData object to transform.
        raw_layer_key (str): Key for the raw expression layer in the AnnData object.
        transformed_layer_key (str): Key for the layer in the AnnData object, where the transformed expression will be saved.
        global_scaling_factor (float): The global scaling factor to apply.
        scaling_dict (Optional[Dict[str, float]]): A dictionary of specific scaling factors for markers.
        inplace (bool): If True, the transformation is applied in place. If False, a new AnnData object is returned.

    Returns
    -------
        Optional[AnnData]: If inplace is False, returns the transformed AnnData object. Otherwise, returns None.
    """
    # combine scaling factors into one dict
    global_dict = {marker: global_scaling_factor for marker in adata.var_names}

    # overwrite scaling factors if dictionary is provided
    if scaling_dict is not None:
        global_dict.update(scaling_dict)

    # apply scaling and arcsinh transformation
    transformed_layer = adata.layers[raw_layer_key].copy() / np.array(list(global_dict.values()))
    adata.layers[transformed_layer_key] = np.arcsinh(transformed_layer)

    if not transform_scatter:
        scatter_prefix = ("FSC", "fsc", "SSC", "ssc")
        is_scatter = [marker.startswith(scatter_prefix) for marker in adata.var_names]

        if any(is_scatter):
            scatter_str = " ,".join(adata.var_names[is_scatter])
            msg = f"Detected scatter features, which are omited for transformation. \nScatter features: {scatter_str}"
            warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)

            adata.layers[transformed_layer_key][:, is_scatter] = adata.layers[raw_layer_key][:, is_scatter]

    return adata if not inplace else None


def logp(
    adata: AnnData,
    raw_layer_key: str = "raw",
    transformed_layer_key: str = "transformed",
    offset: float = 1.0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Apply log transformation to the 'raw' layer of an AnnData object.

    Parameters
    ----------
        adata (AnnData): The AnnData object to normalize.
        raw_layer_key (str): The key of the raw layer to be transformed.
        transformed_layer_key (str): The key to store the transfromed data in `adata.layers`.
        inplace (bool): If True, the normalization is applied in place. If False, a new AnnData object is returned.

    Returns
    -------
        Optional[AnnData]: If inplace is False, returns the normalized AnnData object. Otherwise, returns None.
    """
    adata.layers[transformed_layer_key] = adata.layers[raw_layer_key].copy()
    adata.layers[transformed_layer_key] += offset
    adata.layers[transformed_layer_key] = np.log(adata.layers[transformed_layer_key])

    return adata if not inplace else None


def scale(
    adata: AnnData,
    feature_range: tuple[float, float] = (0.0, 1.0),
    transformed_layer_key: str = "transformed",
    scaled_layer_key: str = "scaled",
    feat_eps: float = 1e-6,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Apply min-max scaling to the transformed layer of an AnnData object.

    Parameters
    ----------
        adata (AnnData): The AnnData object to scale.
        feature_range (Tuple[float, float]): The desired range of the scaled data.
        transformed_layer_key (str): The key of the transformed layer.
        scaled_layer_key (str): The key to store the scaled data in `adata.layers`.
        inplace (bool): If True, the scaling is applied in place. If False, a new AnnData object is returned.

    Returns
    -------
        Optional[AnnData]: If inplace is False, returns the scaled AnnData object. Otherwise, returns None.
    """
    feature_range = (feature_range[0] + feat_eps, feature_range[1] - feat_eps)
    scaler = MinMaxScaler(feature_range=feature_range)
    adata.layers[scaled_layer_key] = scaler.fit_transform(adata.layers[transformed_layer_key].copy())

    scaler_params = {"feature_range": scaler.feature_range, "scale_": scaler.scale_, "min_": scaler.min_}
    adata.uns["scaler_params"] = scaler_params

    return adata if not inplace else None


def register_nan_layer(
    adata: AnnData,
    mask_layer_key: str = "_nan_mask",
    scaled_layer_key: str = "scaled",
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Add a mask layer and replace NaNs by zero in the scaled layer of an AnnData object.

    Parameters
    ----------
        adata (AnnData): The AnnData object to process.
        mask_layer_key (str): The key to store the mask layer in `adata.layers`.
        scaled_layer_key (str): The key of the scaled layer.
        inplace (bool): If True, the processing is applied in place. If False, a new AnnData object is returned.

    Returns
    -------
        Optional[AnnData]: If inplace is False, returns the processed AnnData object. Otherwise, returns None.
    """
    # add mask layer and replace nans by zero
    adata.layers[mask_layer_key] = np.ones_like(adata.layers[scaled_layer_key])
    adata.layers[mask_layer_key][np.isnan(adata.layers[scaled_layer_key])] = 0

    # replace nans by zeroes in expression layer
    adata.layers[scaled_layer_key][np.isnan(adata.layers[scaled_layer_key])] = 0

    return adata if not inplace else None


def merge_batches(
    adata_list: list[AnnData], mask_layer_key: str = "_nan_mask", scaled_layer_key: str = "scaled"
) -> AnnData:
    """
    Merge batches of AnnData objects and handle missing markers.

    Parameters
    ----------
        adata_list (List[AnnData]): List of AnnData objects to be merged.
        mask_layer_key (str): The key to store the mask layer in `adata.layers`.
        scaled_layer_key (str): The key of the scaled layer.

    Returns
    -------
        AnnData: The merged AnnData object.
    """
    # check if there are NaNs before merging
    for batch, adata_batch in enumerate(adata_list):
        if np.isnan(adata_batch.layers[scaled_layer_key]).any():
            error_msg = "Nan values are present in batch {}. This will interfere with downstream processing."
            raise ValueError(error_msg.format(batch))

    adata = ad.concat(adata_list, join="outer", label="batch", fill_value=None)
    all_markers = adata.var_names

    # register missing markers after merging
    if np.isnan(adata.layers[scaled_layer_key]).any():
        backbone_markers = list(all_markers[~np.isnan(adata.layers[scaled_layer_key]).any(axis=0)])
        backbone_str = ", ".join(backbone_markers)

        msg = (
            "Not all proteins are detected across all batches. Will generate nan_layer"
            + f"for imputation of missing proteins. \nBackbone markers: {backbone_str}"
        )
        warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)

        adata = register_nan_layer(
            adata, mask_layer_key=mask_layer_key, scaled_layer_key=scaled_layer_key, inplace=False
        )

        # register which markers are present in which batch
        for batch, adata_batch in enumerate(adata_list):
            batch_name = "_batch" + "_" + str(batch)
            batch_markers = adata_batch.var_names.intersection(all_markers)
            adata.var[batch_name] = all_markers.isin(batch_markers.intersection(all_markers))

    adata.obs_names_make_unique()
    # note: find a way to combine scaling parameters from different batches
    # write scaling params for arcsinh transformation
    # also write the reverse functions

    return adata


def subsample(
    adata: AnnData, n_obs: int = 10000, random_state: int = 0, replace: bool = False, group_by: str = None
) -> Optional[AnnData]:
    """
    Subsample an AnnData object.

    Parameters
    ----------
        adata (AnnData): The AnnData object to downsample.
        n_samples (int): The number of samples to downsample to.
        random_state (int): The random state to use for the downsampling.
        inplace (bool): If True, the downsampling is applied in place. If False, a new AnnData object is returned.

    Returns
    -------
        Optional[AnnData]: If inplace is False, returns the downsampled AnnData object. Otherwise, returns None.
    """
    if group_by is not None:
        if group_by not in adata.obs:
            raise ValueError(f"Group {group_by} not found in adata.obs.")
        group_cats = adata.obs["cell_type"].drop_duplicates().values
        n_obs_group = n_obs // len(group_cats)

        if not replace:
            for group in group_cats:
                if len(adata.obs[adata.obs[group_by] == group]) < n_obs_group:
                    msg = (
                        f"Group {group} has less observations than {n_obs_group} observations."
                        + "Taking all group observations. Set replace to True to sample with replacement."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)

        index = adata.obs.groupby(group_by, as_index=False).apply(
            lambda x: x.sample(n_obs_group, random_state=random_state, replace=replace) if len(x) > n_obs_group else x
        )
        index = index.reset_index()["level_1"].to_list()
        adata_subsampled = adata[index, :].copy()
    else:
        adata_subsampled = adata[adata.obs.sample(n_obs, random_state=random_state, replace=replace).index, :].copy()

    return adata_subsampled
