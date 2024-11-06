from typing import Union

import numpy as np
import pynndescent
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def validate_marker(adata: AnnData, marker: Union[str, list[str]]):
    if isinstance(marker, str):
        marker = [marker]
    for m in marker:
        if m not in adata.var_names:
            raise ValueError(f"Marker {m} not found in adata.var_names.")

def validate_obs_keys(adata: AnnData, obs_key: Union[str, list[str]]):
    if obs_key is not None:
        if isinstance(obs_key, str):
            obs_key = [obs_key]
        for key in obs_key:
            if key is not None:
                if key not in adata.obs:
                    raise ValueError(f"Key {key} not found in adata.obs.")

def validate_obsm_keys(adata, obsm_keys):
    if obsm_keys is not None:
        if isinstance(obsm_keys, str):
            obsm_keys = [obsm_keys]
        for key in obsm_keys:
            if key not in adata.obsm:
                raise KeyError(f"Key '{key}' not found in adata.obs or adata.obsm.")

def validate_layer_key(adata: AnnData, layer_key: str):
    if layer_key is not None:
        if layer_key not in adata.layers:
            raise ValueError(f"Layer key {layer_key} not found in adata.layers.")

def apply_scaling(data, method, feature_range):
    if method == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == "standard":
            scaler = StandardScaler()
    return scaler.fit_transform(data), scaler

def get_n_latent_heuristic(n_vars: int, latent_max: int = 20, latent_min: int = 10):
    n_latent = round(n_vars/2)

    if n_latent > latent_max:
        n_latent = latent_max

    if n_latent < latent_min:
        n_latent = latent_min

    return n_latent

def clip_lfc_factory(min_lfc: float, max_lfc: float):
    def clip_lfc(x, y):
        x = np.clip(x, min_lfc, max_lfc)
        y = np.clip(y, min_lfc, max_lfc)
        return np.log2(x) - np.log2(y)
    return clip_lfc

def validate_expression_range(data, min_exp, max_exp):
    data_in_range =  np.min(data) > min_exp and np.max(data) < max_exp
    return data_in_range

def encode_categories(adata, cat_key):
    """One-Hot encode the categories for the given key."""
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    return ohe.fit_transform(adata.obs[cat_key].values.reshape(-1, 1)), ohe

def impute_with_neighbors(rep_query, rep_ref, cat_encoded_ref, n_neighbors=5):
    """Use pynndescent to find nearest neighbors and impute missing categories."""
    nn_index = pynndescent.NNDescent(rep_ref, n_neighbors=n_neighbors, metric='euclidean')

    # Find the nearest neighbors for query data within the reference data
    indices, distances = nn_index.query(rep_query, k=n_neighbors)

    # Get the neighbor categories for each query point
    neighbor_categories = cat_encoded_ref[indices]  # Shape: (n_query, n_neighbors, n_categories)

    # Sum the one-hot encoded categories for each query point's neighbors
    # This gives the count of each category across the neighbors
    category_sums = np.sum(neighbor_categories, axis=1)  # Shape: (n_query, n_categories)

    # Find the index of the most frequent category for each query point
    imputed_cat_indices = np.argmax(category_sums, axis=1)  # Shape: (n_query,)

    return imputed_cat_indices
