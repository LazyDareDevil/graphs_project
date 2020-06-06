import os
import bz2
import xml.sax
import time
from .osm_content_handler import OSMContentHandler
import pandas as pd

from .utils import log, great_circle_vec, euclidean_dist_vec
from . import settings


def overpass_json_from_file(filename):

    _, ext = os.path.splitext(filename)

    if ext == '.bz2':
        # Use Python 2/3 compatible BZ2File()
        opener = lambda fn: bz2.BZ2File(fn)
    else:
        # Assume an unrecognized file extension is just XML
        opener = lambda fn: open(fn, mode='rb')

    with opener(filename) as file:
        handler = OSMContentHandler()
        xml.sax.parse(file, handler)
        return handler.object


def get_nearest_node(G, point, method='haversine', return_dist=False):
    """
        Return the graph node nearest to some specified (lat, lng) or (y, x) point,
        and optionally the distance between the node and the point. This function
        can use either a haversine or euclidean distance calculator.

        Parameters
        ----------
        G : networkx multidigraph
        point : tuple
            The (lat, lng) or (y, x) point for which we will find the nearest node
            in the graph
        method : str {'haversine', 'euclidean'}
            Which method to use for calculating distances to find nearest node.
            If 'haversine', graph nodes' coordinates must be in units of decimal
            degrees. If 'euclidean', graph nodes' coordinates must be projected.
        return_dist : bool
            Optionally also return the distance (in meters if haversine, or graph
            node coordinate units if euclidean) between the point and the nearest
            node.

        Returns
        -------
        int or tuple of (int, float)
            Nearest node ID or optionally a tuple of (node ID, dist), where dist is
            the distance (in meters if haversine, or graph node coordinate units
            if euclidean) between the point and nearest node
        """

    start_time = time.time()

    if not G or (G.number_of_nodes() == 0):
        raise ValueError('G argument must be not be empty or should contain at least one node')

    # dump graph node coordinates into a pandas dataframe indexed by node id
    # with x and y columns
    coords = [[node, data['x'], data['y']] for node, data in G.nodes(data=True)]
    df = pd.DataFrame(coords, columns=['node', 'x', 'y']).set_index('node')

    # add columns to the dataframe representing the (constant) coordinates of
    # the reference point
    df['reference_y'] = point[0]
    df['reference_x'] = point[1]

    # calculate the distance between each node and the reference point
    if method == 'haversine':
        # calculate distance vector using haversine (ie, for
        # spherical lat-long geometries)
        distances = great_circle_vec(lat1=df['reference_y'],
                                     lng1=df['reference_x'],
                                     lat2=df['y'],
                                     lng2=df['x'])

    elif method == 'euclidean':
        # calculate distance vector using euclidean distances (ie, for projected
        # planar geometries)
        distances = euclidean_dist_vec(y1=df['reference_y'],
                                       x1=df['reference_x'],
                                       y2=df['y'],
                                       x2=df['x'])

    else:
        raise ValueError('method argument must be either "haversine" or "euclidean"')

    # nearest node's ID is the index label of the minimum distance
    nearest_node = distances.idxmin()
    # log('Found nearest node ({}) to point {} in {:,.2f} seconds'.format(nearest_node, point, time.time()-start_time))

    # if caller requested return_dist, return distance between the point and the
    # nearest node as well
    if return_dist:
        return nearest_node, distances.loc[nearest_node]
    else:
        return nearest_node
