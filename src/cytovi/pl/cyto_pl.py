import math
from typing import Union

import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData

from cytovi._utils import check_group_by, check_layer_key, check_marker
from cytovi.pp.cyto_pp import subsample


def histogram(
    adata: ad.AnnData,
    marker: Union[str, list[str]] = "all",
    group_by: str = None,
    layer_key: str = "raw",
    downsample: bool = True,
    n_obs: int = 10000,
    col_wrap=None,
    **kwargs,
):
    """
    Create a FacetGrid of histograms for specified markers in AnnData.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix.

    marker : Union[str, List[str]], optional
        Names of markers to plot. 'all' to plot all markers.

    group_by : str, optional
        Key for grouping or categorizing the data. E.g. key for batch.

    layer_key : str, optional
        Key for the layer in AnnData.

    **kwargs : additional keyword arguments
        Additional arguments to pass to Seaborn's FacetGrid.

    Returns
    -------
    None

    Example:
    ----------
    # Plot density plots for specific markers
    plot_marker_histograms(adata, marker=['Marker1', 'Marker2'], group_by='Condition')

    # Plot density plots for all markers
    plot_marker_histograms(adata, marker='all', group_by='Batch')
    """
    if marker == "all":
        marker = adata.var_names
    elif isinstance(marker, str):
        marker = [marker]

    check_marker(adata, marker)
    check_group_by(adata, group_by)
    check_layer_key(adata, layer_key)

    # subsample if too many observations
    if downsample and adata.n_obs > 10000:
        adata = subsample(adata, n_obs=n_obs, group_by=group_by)

    num_plots = len(marker)

    if col_wrap is None:
        col_wrap = math.ceil(math.sqrt(num_plots))

    data_plot = adata[:, marker].to_df(layer=layer_key)

    if group_by is not None:
        data_plot[group_by] = adata.obs[group_by]

    data_plot_melt = data_plot.melt(id_vars=group_by)

    # generate the plot
    g = sns.FacetGrid(
        data_plot_melt, col="variable", hue=group_by, col_wrap=col_wrap, sharey=False, sharex=False, **kwargs
    )
    g.map(sns.kdeplot, "value", fill=True)
    g.set_titles("{col_name}")
    g.set(yticks=[])
    g.set_axis_labels("", "")
    g.add_legend()
    g.fig.text(0, 0.5, "Density", va="center", ha="center", rotation="vertical")
    plt.show()


def biaxial(
    adata: AnnData,
    marker_x: Union[str, list[str]] = None,
    marker_y: Union[str, list[str]] = None,
    group_by: str = None,
    n_bins: int = 10,
    layer_key: str = "raw",
    downsample: bool = True,
    n_obs: int = 10000,
    **kwargs,
):
    """
    Create a PairGrid of biaxial (scatter and density) plots for specified markers in AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.

    marker_x : Union[str, List[str]], optional
        Variable name(s) to be plotted on the x-axis.

    marker_y : Union[str, List[str]], optional
        Variable name(s) to be plotted on the y-axis.

    group_by : str, optional
        Key for grouping or categorizing the data.

    n_bins : int, optional
        Number of levels for density contours in kdeplot.

    layer_key : str, optional
        Key for the layer in AnnData.

    **kwargs : additional keyword arguments
        Additional arguments to pass to Seaborn's PairGrid.

    Returns
    -------
    None

    Example
    -------
    # Plot biaxial plots for specific markers
    plot_biaxial(adata, marker_x='Marker1', marker_y='Marker2', group_by='Condition')

    # Plot biaxial plots for multiple markers
    plot_biaxial(adata, marker_x=['GeneA', 'GeneB'], marker_y='GeneC', group_by='Batch')
    """
    if isinstance(marker_x, str):
        marker_x = [marker_x]
    if isinstance(marker_y, str):
        marker_y = [marker_y]

    check_marker(adata, marker_x)
    check_marker(adata, marker_y)
    check_group_by(adata, group_by)
    check_layer_key(adata, layer_key)

    # subsample if too many observations
    if downsample and adata.n_obs > 10000:
        adata = subsample(adata, n_obs=n_obs, group_by=group_by)

    # remove marker from marker_x if it is also in marker_y
    if marker_x is not None and marker_y is not None:
        marker_x = list(set(marker_x) - set(marker_y))

    marker = marker_x + marker_y

    data_plot = adata[:, marker].to_df(layer=layer_key)

    if group_by is not None:
        data_plot[group_by] = adata.obs[group_by]

    g = sns.PairGrid(data_plot, x_vars=marker_x, y_vars=marker_y, hue=group_by, **kwargs)
    g.map(sns.kdeplot, levels=n_bins)
    g.map(sns.scatterplot, s=5)
    g.add_legend()
    plt.show()
