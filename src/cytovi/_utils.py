import numpy as np
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def check_marker(adata: AnnData, marker: list[str]):
    for m in marker:
        if m not in adata.var_names:
            raise ValueError(f"Marker {m} not found in adata.var_names.")


def check_group_by(adata: AnnData, group_by: str):
    if group_by is not None:
        if group_by not in adata.obs:
            raise ValueError(f"Group by {group_by} not found in adata.obs.")


def check_layer_key(adata: AnnData, layer_key: str):
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

