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
    groupby: str = None,
    layer_key: str = "raw",
    downsample: bool = True,
    n_obs: int = 10000,
    col_wrap: int = None,
    tight_layout: bool = True,
    save: Union[bool, str] = None,
    return_plot: bool = False,
    kde_kwargs: dict = None,
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
    if kde_kwargs is None:
        kde_kwargs = {}

    if marker == "all":
        marker = adata.var_names
    elif isinstance(marker, str):
        marker = [marker]

    check_marker(adata, marker)
    check_group_by(adata, groupby)
    check_layer_key(adata, layer_key)

    # subsample if too many observations
    if downsample and adata.n_obs > 10000:
        adata = subsample(adata, n_obs=n_obs, groupby=groupby)

    num_plots = len(marker)

    if col_wrap is None:
        col_wrap = math.ceil(math.sqrt(num_plots))

    data_plot = adata[:, marker].to_df(layer=layer_key)

    if groupby is not None:
        data_plot[groupby] = adata.obs[groupby]

    data_plot_melt = data_plot.melt(id_vars=groupby)

    # generate the plot
    g = sns.FacetGrid(
        data_plot_melt, col="variable", hue=groupby, col_wrap=col_wrap, sharey=False, sharex=False, **kwargs
    )
    g.map(sns.kdeplot, "value", fill=True, **kde_kwargs)
    g.set_titles("{col_name}")
    g.set(yticks=[])
    g.set_axis_labels("", "")
    g.add_legend()
    g.fig.text(0, 0.5, "Density", va="center", ha="center", rotation="vertical")

    if tight_layout:
        g.fig.tight_layout()

    if save is not None:
        if save is True:
            save = "marker_histogram.png"
        g.savefig(save)

    if return_plot:
        return g


def biaxial(
    adata: AnnData,
    marker_x: Union[str, list[str]] = None,
    marker_y: Union[str, list[str]] = None,
    color: str = None,
    n_bins: int = 10,
    layer_key: str = "raw",
    downsample: bool = True,
    n_obs: int = 10000,
    sample_color_groups: bool = False,
    save: Union[bool, str] = None,
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
    check_group_by(adata, color)
    check_layer_key(adata, layer_key)

    # subsample if too many observations
    if downsample and adata.n_obs > 10000:
        if color is not None and sample_color_groups is True:
            adata = subsample(adata, n_obs=n_obs, groupby=color)
        else:
            adata = subsample(adata, n_obs=n_obs)

    # remove marker from marker_x if it is also in marker_y
    if marker_x is not None and marker_y is not None:
        marker_x = list(set(marker_x) - set(marker_y))

    marker = marker_x + marker_y

    data_plot = adata[:, marker].to_df(layer=layer_key)

    if color is not None:
        data_plot[color] = adata.obs[color]

    g = sns.PairGrid(data_plot, x_vars=marker_x, y_vars=marker_y, hue=color, **kwargs)
    g.map(sns.kdeplot, levels=n_bins)
    g.map(sns.scatterplot, s=5)
    g.add_legend()

    if save is not None:
        if save is True:
            save = "marker_histogram.png"
        g.savefig(save)

    plt.show()

