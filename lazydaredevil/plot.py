import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

from . import settings
from .save_load import graph_to_gdfs
from .utils import log


def plot_graph(G, bbox=None, fig_height=6, fig_width=None, margin=0.02,
               axis_off=True, equal_aspect=False, bgcolor='w', show=False,
               save=True, close=True, file_format='png', filename='temp',
               dpi=1500, annotate=False, node_color='#66ccff', node_size=0.1,
               node_alpha=0.1, node_edgecolor='none', node_zorder=0.1,
               edge_color='#999999', edge_linewidth=0.5, edge_alpha=0.5,
               use_geom=False):

    log('Begin plotting the graph...')
    node_Xs = [float(x) for _, x in G.nodes(data='x')]
    node_Ys = [float(y) for _, y in G.nodes(data='y')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    # draw the edges as lines from node to node
    start_time = time.time()
    lines = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)
    log('Drew the graph edges in {:,.2f} seconds'.format(time.time()-start_time))

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # if axis_off, turn off the axis display set the margins to zero and point
    # the ticks in so there's no space around the plot
    if axis_off:
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()
    else:
        # if the graph is not projected, conform the aspect ratio to not stretch the plot
        if G.graph['crs'] == settings.default_crs:
            coslat = np.cos((min(node_Ys) + max(node_Ys)) / 2. / 180. * np.pi)
            ax.set_aspect(1. / coslat)
            fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x'], data['y']))

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


def save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off):
    # save the figure if specified
    if save:
        start_time = time.time()

        # create the save folder if it doesn't already exist
        if not os.path.exists(settings.imgs_folder):
            os.makedirs(settings.imgs_folder)
        path_filename = os.path.join(settings.imgs_folder, os.extsep.join([filename, file_format]))

        if file_format == 'svg':
            # if the file_format is svg, prep the fig/ax a bit for saving
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            ax.patch.set_alpha(0.)
            fig.patch.set_alpha(0.)
            fig.savefig(path_filename, bbox_inches=0, format=file_format, facecolor=fig.get_facecolor(), transparent=True)
        else:
            if axis_off:
                # if axis is turned off, constrain the saved figure's extent to
                # the interior of the axis
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            else:
                extent = 'tight'
            fig.savefig(path_filename, dpi=dpi, bbox_inches=extent, format=file_format, facecolor=fig.get_facecolor(), transparent=True)
        log('Saved the figure to disk in {:,.2f} seconds'.format(time.time()-start_time))

    # show the figure if specified
    if show:
        start_time = time.time()
        plt.show()
        log('Showed the plot in {:,.2f} seconds'.format(time.time()-start_time))
    # if show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()

    return fig, ax


def node_list_to_coordinate_lines(G, node_list, use_geom=True):
    """
    Given a list of nodes, return a list of lines that together follow the path
    defined by the list of nodes.

    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node

    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start), (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    return lines


def plot_graph_route(G, route, bbox=None, fig_height=6, fig_width=None,
                     margin=0.02, bgcolor='w', axis_off=True, show=False,
                     save=True, close=True, file_format='png', filename='temp',
                     dpi=1000, annotate=False, node_color='#999999',
                     node_size=0.1, node_alpha=0.1, node_edgecolor='none',
                     node_zorder=0.1, edge_color='#999999', edge_linewidth=1,
                     edge_alpha=0.5, use_geom=True, origin_point=None,
                     destination_point=None, route_color='r', route_linewidth=0.5,
                     route_alpha=0.2, orig_dest_node_alpha=0.5,
                     orig_dest_node_size=1, orig_dest_node_color='black',
                     orig_dest_point_color='black'):
    """
    Plot a route along a networkx spatial graph.

    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes
    orig_dest_point_color : string
        the color of the origin and destination points if being plotted instead
        of nodes

    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the route
    fig, ax = plot_graph(G, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                         margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                         show=False, save=False, close=False, filename=filename,
                         dpi=dpi, annotate=annotate, node_color=node_color,
                         node_size=node_size, node_alpha=node_alpha,
                         node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                         edge_color=edge_color, edge_linewidth=edge_linewidth,
                         edge_alpha=edge_alpha, use_geom=use_geom)

    # the origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]

    if origin_point is None or destination_point is None:
        # if caller didn't pass points, use the first and last node in route as
        # origin/destination
        origin_destination_lats = (G.nodes[origin_node]['y'], G.nodes[destination_node]['y'])
        origin_destination_lons = (G.nodes[origin_node]['x'], G.nodes[destination_node]['x'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_lats = (origin_point[0], destination_point[0])
        origin_destination_lons = (origin_point[1], destination_point[1])
        orig_dest_node_color = orig_dest_point_color

    # scatter the origin and destination points
    ax.scatter(origin_destination_lons, origin_destination_lats, s=orig_dest_node_size,
               c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the route lines
    lines = node_list_to_coordinate_lines(G, route, use_geom)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


def plot_graph_routes(G, routes, bbox=None, fig_height=10, fig_width=None,
                      margin=0.02, bgcolor='w', axis_off=True, show=False,
                      save=True, close=True, file_format='png', filename='temp',
                      dpi=800, annotate=False, node_color='#999999',
                      node_size=0.1, node_alpha=0.1, node_edgecolor='none',
                      node_zorder=0.1, edge_color='#999999', edge_linewidth=0.5,
                      edge_alpha=1, use_geom=True, orig_dest_points=None,
                      route_color='r', route_linewidth=1,
                      route_alpha=1, orig_dest_node_alpha=1,
                      orig_dest_node_size=10, orig_dest_node_color='black',
                      orig_dest_point_color='black'):
    """
    Plot several routes along a networkx spatial graph.

    Parameters
    ----------
    G : networkx multidigraph
    routes : list
        the routes as a list of lists of nodes
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    orig_dest_points : list of tuples
        optional, a group of (lat, lon) points to plot instead of the
        origins and destinations of each route nodes
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes
    orig_dest_point_color : string
        the color of the origin and destination points if being plotted instead
        of nodes

    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the routes
    fig, ax = plot_graph(G, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                         margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                         show=False, save=False, close=False, filename=filename,
                         dpi=dpi, annotate=annotate, node_color=node_color,
                         node_size=node_size, node_alpha=node_alpha,
                         node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                         edge_color=edge_color, edge_linewidth=edge_linewidth,
                         edge_alpha=edge_alpha, use_geom=use_geom)

    # save coordinates of the given reference points
    orig_dest_points_lats = []
    orig_dest_points_lons = []
    start_point_lat = 0
    start_point_lon = 0
    start_point_color = '#9b00a1'
    if orig_dest_points is None:
        # if caller didn't pass points, use the first and last node in each route as
        # origin/destination points
        for route in routes:
            if routes.index(route) == 0:
                sp = route[0]
                start_point_lat = G.nodes[sp]['y']
                start_point_lon = G.nodes[sp]['x']
            origin_node = route[0]
            destination_node = route[-1]
            orig_dest_points_lats.append(G.nodes[origin_node]['y'])
            orig_dest_points_lats.append(G.nodes[destination_node]['y'])
            orig_dest_points_lons.append(G.nodes[origin_node]['x'])
            orig_dest_points_lons.append(G.nodes[destination_node]['x'])

    else:
        # otherwise, use the passed points as origin/destination points
        for point in orig_dest_points:
            orig_dest_points_lats.append(point[0])
            orig_dest_points_lons.append(point[1])
        orig_dest_node_color = orig_dest_point_color

    # scatter the origin and destination points
    ax.scatter(orig_dest_points_lons, orig_dest_points_lats, s=orig_dest_node_size,
               c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)
    ax.scatter(start_point_lon, start_point_lat, s=orig_dest_node_size*5,
               c=start_point_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the routes lines
    lines = []
    for route in routes:
        lines.extend(node_list_to_coordinate_lines(G, route, use_geom))

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


def plot_trees(G, trees):
    p = []
    for poi in trees.keys():
        for key in trees[poi].keys():
            for path in trees[poi][key]:
                p.append(path)
        plot_graph_routes(G, p, filename='tree_'+str(poi))
        p = []


def plot_trees_clusters(G, trees, clusters, cpois, poi):
    p = []
    for cluster in trees:
        for key in trees[cluster].keys():
            for path in trees[cluster][key]:
                p.append(path)
        plot_graph_routes_clusters(G, p, clusters=clusters[cluster], cpois=cpois[cluster],
                                   filename=str(cluster)+'_clusters_tree_'+str(poi['id']))
        p = []


def plot_graph_routes_clusters(G, routes, bbox=None, fig_height=10, fig_width=None,
                               margin=0.02, bgcolor='w', axis_off=True, show=False,
                               save=True, close=True, file_format='png', filename='temp',
                               dpi=800, annotate=False, node_color='#999999',
                               node_size=0.1, node_alpha=1, node_edgecolor='none',
                               node_zorder=0.1, edge_color='#999999', edge_linewidth=0.5,
                               edge_alpha=1, use_geom=True, orig_dest_points=None,
                               route_color='r', route_linewidth=1,
                               route_alpha=1, orig_dest_node_alpha=1,
                               orig_dest_node_size=10, orig_dest_node_color='black',
                               orig_dest_point_color='black',
                               clusters=[], cpois=[],
                               cluster_points_colors=['#997903', '#0a44c2', '#23c20a', '#997903', '#850518'],
                               cpois_color='#370147'):

    # plot the graph but not the routes
    fig, ax = plot_graph(G, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                             margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                             show=False, save=False, close=False, filename=filename,
                             dpi=dpi, annotate=annotate, node_color=node_color,
                             node_size=node_size, node_alpha=node_alpha,
                             node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                             edge_color=edge_color, edge_linewidth=edge_linewidth,
                             edge_alpha=edge_alpha, use_geom=use_geom)

    # save coordinates of the given reference points
    orig_dest_points_lats = []
    orig_dest_points_lons = []
    start_point_lat = 0
    start_point_lon = 0
    start_point_color = '#9b00a1'
    if orig_dest_points is None:
        # if caller didn't pass points, use the first and last node in each route as
        # origin/destination points
        for route in routes:
            if routes.index(route) == 0:
                sp = route[0]
                start_point_lat = G.nodes[sp]['y']
                start_point_lon = G.nodes[sp]['x']
            origin_node = route[0]
            destination_node = route[-1]
            orig_dest_points_lats.append(G.nodes[origin_node]['y'])
            orig_dest_points_lats.append(G.nodes[destination_node]['y'])
            orig_dest_points_lons.append(G.nodes[origin_node]['x'])
            orig_dest_points_lons.append(G.nodes[destination_node]['x'])

    else:
        # otherwise, use the passed points as origin/destination points
        for point in orig_dest_points:
            orig_dest_points_lats.append(point[0])
            orig_dest_points_lons.append(point[1])
        orig_dest_node_color = orig_dest_point_color

    # scatter the origin and destination points
    ax.scatter(orig_dest_points_lons, orig_dest_points_lats, s=orig_dest_node_size,
               c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)
    i = 0
    for cluster in clusters:
        for el in cluster:
            point = G.nodes[el]
            ax.scatter(point['x'], point['y'], s=orig_dest_node_size*5,
                       c=cluster_points_colors[i], alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)
        i += 1
        if i > 5:
            i = 0

    for cpoi in cpois:
        ax.scatter(cpoi['x'], cpoi['y'], s=orig_dest_node_size*5,
                   c=cpois_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    ax.scatter(start_point_lon, start_point_lat, s=orig_dest_node_size*5,
               c=start_point_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)
    # plot the routes lines
    lines = []
    for route in routes:
        lines.extend(node_list_to_coordinate_lines(G, route, use_geom))

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax
