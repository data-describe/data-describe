import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import geoplot
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mwdata.utilities.contextmanager import _context_manager
from mwdata import load_data


@_context_manager
def maps(data, map_type='choropleth', color=None, choropleth_kwargs=None, kde_kwargs=None, context=None):
    """ Mapping for geospatial data

    Args:
        data: Geopandas data frame or shapefile path
        map_type: Default = 'choropleth'
            choropleth: Thematic map where areas are shaded in proportion to a statistical variable
            kde: Spatial density estimate plot for the distribution of input data
        color: Color map by a numeric column
        choropleth_kwargs: Key word arguments for geopandas.plot
        kde_kwargs: Key word arguments for geoplot.kdeplot
        context: The context

    Returns:
        fig: Choropleth map or KDE map
    """
    if not isinstance(data, gpd.geodataframe.GeoDataFrame):
        data = load_data(data)
        if not isinstance(data, gpd.geodataframe.GeoDataFrame):
            raise NotImplementedError('Shapefile required')
        
    if map_type == 'choropleth':
        fig = choropleth(data, color, choropleth_kwargs=choropleth_kwargs, context=context)

    elif map_type == 'kde':
        if data.shape[0] > 100:
            warnings.warn("Large number of rows, processing will take awhile")
        fig = kde_map(data, kde_kwargs=kde_kwargs, context=context)

    else:
        raise ValueError("{} map is not supported".format(map_type))
    return fig


@_context_manager
def choropleth(data, color=None, choropleth_kwargs=None, context=None):
    """ Thematic map where areas are shaded in proportion to a statistical variable

    Args:
        data: Geopandas data frame
        color: Color map by a numeric column
        choropleth_kwargs: Key word arguments geopandas.plot
        context: The context

    Returns:
        fig: Choropleth map
    """
    if choropleth_kwargs is None:
        choropleth_kwargs = {}

    fig, ax = plt.subplots(1, 1, figsize=(context.fig_width, context.fig_height))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Choropleth Map")

    if color is not None:
        if np.issubdtype(data[color].dtype, np.number):
            legend = True
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="17%", pad=0.1)
            cax.set_title(color)

        else:
            raise ValueError('{} should be numeric type'.format(color))
    else:
        legend = False
        cax = None

    fig = data.plot(column=color,
                    legend=legend,
                    ax=ax,
                    cax=cax,
                    **choropleth_kwargs)
    return fig


@_context_manager
def kde_map(data, kde_kwargs=None, context=None):
    """ Spatial density estimate plot for the distribution of input data

    Args:
        data: Geopandas data frame
        kde_kwargs: Key word arguments for geoplot.kdeplot
        context: The context

    Returns:
        fig: KDE plot
    """
    if kde_kwargs is None:
        kde_kwargs = dict()
        kde_kwargs.setdefault('alpha', 0.7)
    else:
        kde_kwargs.setdefault('alpha', 0.7)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module='scipy',
                                message=r"Using a non-tuple sequence for multidimensional indexing is deprecated")
        ax = geoplot.kdeplot(df=data.geometry.centroid,
                         figsize=(context.fig_width, context.fig_height),
                         clip=data.dissolve('state').geometry,
                         shade_lowest=False,
                         cmap='viridis',
                         shade=True,
                         **kde_kwargs)

        ax.set_title("KDE Plot")

        fig = geoplot.polyplot(data, ax=ax)

    return fig
