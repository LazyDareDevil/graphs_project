import sys
import os
import datetime as dt
import unicodedata
import numpy as np
import logging as lg
import networkx as nx
from . import settings


def make_str(value):
    # python 3.x has no unicode type, so if error, use str type
    return str(value)


def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """
    Vectorized function to calculate the great-circle distance between two
    points or between vectors of points, using haversine.

    Parameters
    ----------
    lat1 : float or array of float
    lng1 : float or array of float
    lat2 : float or array of float
    lng2 : float or array of float
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is
        meters)

    Returns
    -------
    distance : float or vector of floats
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in
        units of earth_radius
    """

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    d_phi = phi2 - phi1

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)
    d_theta = theta2 - theta1

    h = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_theta / 2) ** 2
    h = np.minimum(1.0, h)  # protect against floating point errors

    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    distance = arc * earth_radius
    return distance


def euclidean_dist_vec(y1, x1, y2, x2):
    """
    Vectorized function to calculate the euclidean distance between two points
    or between vectors of points.

    Parameters
    ----------
    y1 : float or array of float
    x1 : float or array of float
    y2 : float or array of float
    x2 : float or array of float

    Returns
    -------
    distance : float or array of float
        distance or vector of distances from (x1, y1) to (x2, y2) in graph units
    """

    # euclid's formula
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance


def log(message, level=None, name=None, filename=None):
    """
    Write a message to the log file and/or print to the the console.

    Parameters
    ----------
    message : string
        the content of the message to log
    level : int
        one of the logger.level constants
    name : string
        name of the logger
    filename : string
        name of the log file

    Returns
    -------
    None
    """

    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    # if logging to file is turned on
    if settings.log_file:
        # get the current logger (or create a new one, if none), then log
        # message at requested level
        logger = get_logger(level=level, name=name, filename=filename)
        if level ==  lg.DEBUG:
            logger.debug(message)
        elif level == lg.INFO:
            logger.info(message)
        elif level == lg.WARNING:
            logger.warning(message)
        elif level == lg.ERROR:
            logger.error(message)

    # if logging to console is turned on, convert message to ascii and print to
    # the console
    if settings.log_console:
        # capture current stdout, then switch it to the console, print the
        # message, then switch back to what had been the stdout. this prevents
        # logging to notebook - instead, it goes to console
        standard_out = sys.stdout
        sys.stdout = sys.__stdout__

        # convert message to ascii for console display so it doesn't break
        # windows terminals
        message = unicodedata.normalize('NFKD', make_str(message)).encode('ascii', errors='replace').decode()
        print(message)
        sys.stdout = standard_out


def get_logger(level=None, name=None, filename=None):
    """
    Create a logger or return the current one if already instantiated.

    Parameters
    ----------
    level : int
        one of the logger.level constants
    name : string
        name of the logger
    filename : string
        name of the log file

    Returns
    -------
    logger.logger
    """

    if level is None:
        level = settings.log_level
    if name is None:
        name = settings.log_name
    if filename is None:
        filename = settings.log_filename

    logger = lg.getLogger(name)

    # if a logger with this name is not already set up
    if not getattr(logger, 'handler_set', None):

        # get today's date and construct a log filename
        todays_date = dt.datetime.today().strftime('%Y_%m_%d')
        log_filename = os.path.join(settings.logs_folder, '{}_{}.log'.format(filename, todays_date))

        # if the logs folder does not already exist, create it
        if not os.path.exists(settings.logs_folder):
            os.makedirs(settings.logs_folder)

        # create file handler and log formatter and set them up
        handler = lg.FileHandler(log_filename, encoding='utf-8')
        formatter = lg.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.handler_set = True

    return logger


def get_unique_nodes_ordered_from_way(way_edges_df):
    """
    Function to recover the original order of nodes from a dataframe
    of edges associated with a single OSM way.

    Parameters
    ----------
    way_edges_df : pandas.DataFrame()
        Dataframe containing columns 'u' and 'v' corresponding to
        origin/desitination nodes.

    Returns
    -------
    unique_ordered_nodes : list
        An ordered list of unique node IDs

    NOTE: If the edges do not all connect (e.g. [(1, 2), (2,3),
    (10, 11), (11, 12), (12, 13)]), then this method will return
    only those nodes associated with the largest component of
    connected edges, even if subsequent connected chunks are contain
    more total nodes. This is done to ensure a proper topological
    representation of nodes in the XML way records because if there
    are unconnected components, the sorting algorithm cannot recover
    their original order. I don't believe that we would ever encounter
    this kind of disconnected structure of nodes within a given way,
    but as best I could tell it is not explicitly forbidden in the
    OSM XML design schema. I'm using a print statement right now to
    tell the user whether or not any nodes have been dropped and
    how many.
    """

    G = nx.MultiDiGraph()
    all_nodes = list(way_edges_df['u'].values) + \
        list(way_edges_df['v'].values)

    G.add_nodes_from(all_nodes)
    G.add_edges_from(way_edges_df[['u', 'v']].values)
    wccs = nx.weakly_connected_components(G)
    largest_wcc = max(wccs, key=len)
    node_subset = set(largest_wcc)

    # NOTE: this code (L387-403) is copied from geo_utils.py
    # which cannot be imported here without triggering a
    # circular import error. This should be fixed next time the
    # code base is refactored

    # copy nodes into new graph
    G2 = G.__class__()
    G2.add_nodes_from((n, G.nodes[n]) for n in node_subset)

    # copy edges to new graph, including parallel edges
    if G2.is_multigraph:
        G2.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, keydict in nbrs.items() if nbr in node_subset
            for key, d in keydict.items())
    else:
        G2.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, d in nbrs.items() if nbr in node_subset)

    # update graph attribute dict, and return graph
    G2.graph.update(G.graph)

    unique_ordered_nodes = list(nx.topological_sort(G2))
    num_unique_nodes = len(np.unique(all_nodes))

    if len(unique_ordered_nodes) < num_unique_nodes:
        print('Recovered order for {0} of {1} nodes'.format(
            len(unique_ordered_nodes), num_unique_nodes))

    return unique_ordered_nodes
