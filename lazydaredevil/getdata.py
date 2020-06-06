from .download import osm_polygon_download, overpass_request, get_osm_filter
from .projection import project_gdf, project_geometry
from .utils import *
from .errors import *

import geopandas as gpd
import logging as lg
import math
import networkx as nx
import numpy as np
import pandas as pd
import random
import time
from OSMPythonTools.overpass import *
from OSMPythonTools.nominatim import *
import urllib.error

from itertools import groupby
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .geo_utils import overpass_json_from_file


def create_graph(response_jsons, name='unnamed', bidirectional=False):
    log('Creating networkx graph from downloaded OSM data...')
    start_time = time.time()

    # make sure we got data back from the server requests
    elements = []
    for response_json in response_jsons:
        elements.extend(response_json['elements'])
    if len(elements) < 1:
        raise EmptyOverpassResponse('There are no data elements in the response JSON objects')

    # create the graph as a MultiDiGraph and set the original CRS to default_crs
    G = nx.MultiDiGraph(name=name, crs=settings.default_crs)

    # extract nodes and paths from the downloaded osm data
    nodes = {}
    paths = {}
    for osm_data in response_jsons:
        nodes_temp, paths_temp = parse_osm_nodes_paths(osm_data)
        for key, value in nodes_temp.items():
            nodes[key] = value
        for key, value in paths_temp.items():
            paths[key] = value

    # add each osm node to the graph
    for node, data in nodes.items():
        G.add_node(node, **data)

    # add each osm way (aka, path) to the graph
    G = add_paths(G, paths, bidirectional=bidirectional)

    log('Created graph with {:,} nodes and {:,} edges in {:,.2f} seconds'.format(len(list(G.nodes())),
                                                                                 len(list(G.edges())),
                                                                                 time.time() - start_time))

    # add length (great circle distance between nodes) attribute to each edge to
    # use as weight
    if len(G.edges) > 0:
        G = add_edge_lengths(G)

    return G


def add_edge_lengths(G):
    start_time = time.time()

    # first load all the edges' origin and destination coordinates as a
    # dataframe indexed by u, v, key

    coords = np.array([[u, v, k, G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']] for u, v, k in
                       G.edges(keys=True)])
    df_coords = pd.DataFrame(coords, columns=['u', 'v', 'k', 'u_y', 'u_x', 'v_y', 'v_x'])
    df_coords[['u', 'v', 'k']] = df_coords[['u', 'v', 'k']].astype(np.int64)
    df_coords = df_coords.set_index(['u', 'v', 'k'])

    # then calculate the great circle distance with the vectorized function
    gc_distances = great_circle_vec(lat1=df_coords['u_y'],
                                    lng1=df_coords['u_x'],
                                    lat2=df_coords['v_y'],
                                    lng2=df_coords['v_x'])

    # fill nulls with zeros and round to the millimeter
    gc_distances = gc_distances.fillna(value=0).round(3)
    nx.set_edge_attributes(G, name='length', values=gc_distances.to_dict())

    log('Added edge lengths to graph in {:,.2f} seconds'.format(time.time() - start_time))
    return G


def parse_osm_nodes_paths(osm_data):
    nodes = {}
    paths = {}
    for element in osm_data['elements']:
        if element['type'] == 'node':
            key = element['id']
            nodes[key] = get_node(element)
        elif element['type'] == 'way':  # osm calls network paths 'ways'
            key = element['id']
            paths[key] = get_path(element)

    return nodes, paths


def get_node(element):
    node = {}
    node['y'] = element['lat']
    node['x'] = element['lon']
    node['osmid'] = element['id']
    if 'tags' in element:
        for useful_tag in settings.useful_tags_node:
            if useful_tag in element['tags']:
                node[useful_tag] = element['tags'][useful_tag]
    return node


def get_path(element):
    path = {}
    path['osmid'] = element['id']

    # remove any consecutive duplicate elements in the list of nodes
    grouped_list = groupby(element['nodes'])
    path['nodes'] = [group[0] for group in grouped_list]

    if 'tags' in element:
        for useful_tag in settings.useful_tags_path:
            if useful_tag in element['tags']:
                path[useful_tag] = element['tags'][useful_tag]
    return path


def add_path(G, data, one_way):
    # extract the ordered list of nodes from this path element, then delete it
    # so we don't add it as an attribute to the edge later
    path_nodes = data['nodes']
    del data['nodes']

    # set the oneway attribute to the passed-in value, to make it consistent
    # True/False values, but only do this if you aren't forcing all edges to
    # oneway with the all_oneway setting. With the all_oneway setting, you
    # likely still want to preserve the original OSM oneway attribute.
    if not settings.all_oneway:
        data['oneway'] = one_way

    # zip together the path nodes so you get tuples like (0,1), (1,2), (2,3)
    # and so on
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    G.add_edges_from(path_edges, **data)

    # if the path is NOT one-way
    if not one_way:
        # reverse the direction of each edge and add this path going the
        # opposite direction
        path_edges_opposite_direction = [(v, u) for u, v in path_edges]
        G.add_edges_from(path_edges_opposite_direction, **data)


def add_paths(G, paths, bidirectional=False):
    # the list of values OSM uses in its 'oneway' tag to denote True
    # updated list of of values OSM uses based on https://www.geofabrik.de/de/data/geofabrik-osm-gis-standard-0.7.pdf
    osm_oneway_values = ['yes', 'true', '1', '-1', 'T', 'F']

    for data in paths.values():

        if settings.all_oneway is True:
            add_path(G, data, one_way=True)
        # if this path is tagged as one-way and if it is not a walking network,
        # then we'll add the path in one direction only
        elif ('oneway' in data and data['oneway'] in osm_oneway_values) and not bidirectional:
            if data['oneway'] == '-1' or data['oneway'] == 'T':
                # paths with a one-way value of -1 or T are one-way, but in the
                # reverse direction of the nodes' order, see osm documentation
                data['nodes'] = list(reversed(data['nodes']))
            # add this path (in only one direction) to the graph
            add_path(G, data, one_way=True)

        elif ('junction' in data and data['junction'] == 'roundabout') and not bidirectional:
            # roundabout are also oneway but not tagged as is
            add_path(G, data, one_way=True)

        # else, this path is not tagged as one-way or it is a walking network
        # (you can walk both directions on a one-way street)
        else:
            # add this path (in both directions) to the graph and set its
            # 'oneway' attribute to False. if this is a walking network, this
            # may very well be a one-way street (as cars/bikes go), but in a
            # walking-only network it is a bi-directional edge
            add_path(G, data, one_way=False)

    return G


def osm_net_download(polygon=None,
                     network_type='all_private', timeout=180, memory=None,
                     max_query_area_size=50 * 1000 * 50 * 1000, infrastructure='way["highway"]'):
    # check if we're querying by polygon or by bounding box based on which
    # argument(s) where passed into this function
    by_poly = polygon
    if not by_poly:
        raise InsufficientNetworkQueryArguments(
            'You must pass a polygon')

    response_jsons = []

    # pass server memory allocation in bytes for the query to the API
    # if None, pass nothing so the server will use its default allocation size
    # otherwise, define the query's maxsize parameter value as whatever the
    # caller passed in
    if memory is None:
        maxsize = ''
    else:
        maxsize = '[maxsize:{}]'.format(memory)
    osm_filter = get_osm_filter(network_type)
    overpass_settings = settings.default_overpass_query_settings.format(timeout=timeout, maxsize=maxsize)
    # define the query to send the API
    # specifying way["highway"] means that all ways returned must have a highway
    # key. the {filters} then remove ways by key/value. the '>' makes it recurse
    # so we get ways and way nodes. maxsize is in bytes.

    if by_poly:
        # project to utm, divide polygon up into sub-polygons if area exceeds a
        # max size (in meters), project back to lat-long, then get a list of
        # polygon(s) exterior coordinates
        geometry_proj, crs_proj = project_geometry(polygon)
        geometry_proj_consolidated_subdivided = consolidate_subdivide_geometry(geometry_proj,
                                                                               max_query_area_size=max_query_area_size)
        geometry, _ = project_geometry(geometry_proj_consolidated_subdivided, crs=crs_proj, to_latlong=True)
        polygon_coord_strs = get_polygons_coordinates(geometry)
        log('Requesting network data within polygon from API in {:,} request(s)'.format(len(polygon_coord_strs)))
        start_time = time.time()

        # pass each polygon exterior coordinates in the list to the API, one at
        # a time
        for polygon_coord_str in polygon_coord_strs:
            query_template = '{settings};({infrastructure}{filters}(poly:"{polygon}");>;);out;'
            query_str = query_template.format(polygon=polygon_coord_str, filters=osm_filter,
                                              infrastructure=infrastructure, settings=overpass_settings)
            response_json = overpass_request(data={'data': query_str}, timeout=timeout)
            response_jsons.append(response_json)
        log('Got all network data within polygon from API in {:,} request(s) and {:,.2f} seconds'.format(
            len(polygon_coord_strs), time.time() - start_time))

    return response_jsons


def get_polygons_coordinates(geometry):
    # extract the exterior coordinates of the geometry to pass to the API later
    polygons_coords = []
    if isinstance(geometry, Polygon):
        x, y = geometry.exterior.xy
        polygons_coords.append(list(zip(x, y)))
    elif isinstance(geometry, MultiPolygon):
        for polygon in geometry:
            x, y = polygon.exterior.xy
            polygons_coords.append(list(zip(x, y)))
    else:
        raise TypeError('Geometry must be a shapely Polygon or MultiPolygon')

    # convert the exterior coordinates of the polygon(s) to the string format
    # the API expects
    polygon_coord_strs = []
    for coords in polygons_coords:
        s = ''
        separator = ' '
        for coord in list(coords):
            # round floating point lats and longs to 6 decimal places (ie, ~100 mm),
            # so we can hash and cache strings consistently
            s = '{}{}{:.6f}{}{:.6f}'.format(s, separator, coord[1], separator, coord[0])
        polygon_coord_strs.append(s.strip(separator))

    return polygon_coord_strs


def consolidate_subdivide_geometry(geometry, max_query_area_size):
    # let the linear length of the quadrats (with which to subdivide the
    # geometry) be the square root of max area size
    quadrat_width = math.sqrt(max_query_area_size)

    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise TypeError('Geometry must be a shapely Polygon or MultiPolygon')

    # if geometry is a MultiPolygon OR a single Polygon whose area exceeds the
    # max size, get the convex hull around the geometry
    if isinstance(geometry, MultiPolygon) or (isinstance(geometry, Polygon) and geometry.area > max_query_area_size):
        geometry = geometry.convex_hull

    # if geometry area exceeds max size, subdivide it into smaller sub-polygons
    if geometry.area > max_query_area_size:
        geometry = quadrat_cut_geometry(geometry, quadrat_width=quadrat_width)

    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])

    return geometry


def quadrat_cut_geometry(geometry, quadrat_width, min_num=3, buffer_amount=1e-9):
    # create n evenly spaced points between the min and max x and y bounds
    west, south, east, north = geometry.bounds
    x_num = math.ceil((east - west) / quadrat_width) + 1
    y_num = math.ceil((north - south) / quadrat_width) + 1
    x_points = np.linspace(west, east, num=max(x_num, min_num))
    y_points = np.linspace(south, north, num=max(y_num, min_num))

    # create a quadrat grid of lines at each of the evenly spaced points
    vertical_lines = [LineString([(x, y_points[0]), (x, y_points[-1])]) for x in x_points]
    horizont_lines = [LineString([(x_points[0], y), (x_points[-1], y)]) for y in y_points]
    lines = vertical_lines + horizont_lines

    # buffer each line to distance of the quadrat width divided by 1 billion,
    # take their union, then cut geometry into pieces by these quadrats
    buffer_size = quadrat_width * buffer_amount
    lines_buffered = [line.buffer(buffer_size) for line in lines]
    quadrats = unary_union(lines_buffered)
    multipoly = geometry.difference(quadrats)

    return multipoly


def graph_from_polygon(polygon, network_type='all_private', truncate_by_edge=False, name='unnamed',
                       timeout=180, memory=None, max_query_area_size=50 * 1000 * 50 * 1000,
                       infrastructure='way["highway"]'):
    # verify that the geometry is valid and is a shapely Polygon/MultiPolygon
    # before proceeding
    if not polygon.is_valid:
        raise TypeError('Shape does not have a valid geometry')
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError('Geometry must be a shapely Polygon or MultiPolygon. If you requested '
                        'graph from place name or address, make sure your query resolves to a '
                        'Polygon or MultiPolygon, and not some other geometry, like a Point. '
                        'See OSMnx documentation for details.')

    # download a list of API responses for the polygon/multipolygon
    response_jsons = osm_net_download(polygon=polygon, network_type=network_type,
                                      timeout=timeout, memory=memory,
                                      max_query_area_size=max_query_area_size,
                                      infrastructure=infrastructure)

    # create the graph from the downloaded data
    G = create_graph(response_jsons, name=name, bidirectional=network_type in settings.bidirectional_network_types)

    # truncate the graph to the extent of the polygon
    G = truncate_graph_polygon(G, polygon, truncate_by_edge=truncate_by_edge)

    # simplify the graph topology as the last step. don't truncate after
    # simplifying or you may have simplified out to an endpoint beyond the
    # truncation distance, in which case you will then strip out your entire
    # edge

    log('graph_from_polygon() returning graph with {:,} nodes and {:,} edges'.format(len(list(G.nodes())),
                                                                                     len(list(G.edges()))))
    return G


def intersect_index_quadrats(gdf, geometry, quadrat_width=0.05, min_num=3, buffer_amount=1e-9):
    # create an empty dataframe to append matches to
    points_within_geometry = pd.DataFrame()

    # cut the geometry into chunks for r-tree spatial index intersecting
    multipoly = quadrat_cut_geometry(geometry, quadrat_width=quadrat_width, buffer_amount=buffer_amount,
                                     min_num=min_num)

    # create an r-tree spatial index for the nodes (ie, points)
    start_time = time.time()
    sindex = gdf['geometry'].sindex
    log('Created r-tree spatial index for {:,} points in {:,.2f} seconds'.format(len(gdf), time.time() - start_time))

    # loop through each chunk of the geometry to find approximate and then
    # precisely intersecting points
    start_time = time.time()
    for poly in multipoly:

        # buffer by the tiny distance to account for any space lost in the
        # quadrat cutting, otherwise may miss point(s) that lay directly on
        # quadrat line
        buffer_size = quadrat_width * buffer_amount
        poly = poly.buffer(buffer_size).buffer(0)

        # find approximate matches with r-tree, then precise matches from those
        # approximate ones
        if poly.is_valid and poly.area > 0:
            possible_matches_index = list(sindex.intersection(poly.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(poly)]
            points_within_geometry = points_within_geometry.append(precise_matches)

    if len(points_within_geometry) > 0:
        # drop duplicate points, if buffered poly caused an overlap on point(s)
        # that lay directly on a quadrat line
        points_within_geometry = points_within_geometry.drop_duplicates(subset='node')
    else:
        # after simplifying the graph, and given the requested network type,
        # there are no nodes inside the polygon - can't create graph from that
        # so throw error
        raise Exception('There are no nodes within the requested geometry')

    log('Identified {:,} nodes inside polygon in {:,.2f} seconds'.format(len(points_within_geometry),
                                                                         time.time() - start_time))
    return points_within_geometry


def truncate_graph_polygon(G, polygon, truncate_by_edge=False, quadrat_width=0.05, min_num=3, buffer_amount=1e-9):
    start_time = time.time()
    G = G.copy()
    log('Identifying all nodes that lie outside the polygon...')

    # get a GeoDataFrame of all the nodes
    node_geom = [Point(data['x'], data['y']) for _, data in G.nodes(data=True)]
    gdf_nodes = gpd.GeoDataFrame({'node': list(G.nodes()), 'geometry': node_geom})
    gdf_nodes.crs = G.graph['crs']

    # find all the nodes in the graph that lie outside the polygon
    points_within_geometry = intersect_index_quadrats(gdf_nodes, polygon, quadrat_width=quadrat_width, min_num=min_num,
                                                      buffer_amount=buffer_amount)
    nodes_outside_polygon = gdf_nodes[~gdf_nodes.index.isin(points_within_geometry.index)]

    if truncate_by_edge:
        nodes_to_remove = []
        for node in nodes_outside_polygon['node']:
            neighbors = pd.Series(list(G.successors(node)) + list(G.predecessors(node)))
            # check if all the neighbors of this node also lie outside polygon
            if neighbors.isin(nodes_outside_polygon['node']).all():
                nodes_to_remove.append(node)
    else:
        nodes_to_remove = nodes_outside_polygon['node']

    # now remove from the graph all those nodes that lie outside the place
    # polygon
    G.remove_nodes_from(nodes_to_remove)
    log('Removed {:,} nodes outside polygon in {:,.2f} seconds'.format(len(nodes_outside_polygon),
                                                                       time.time() - start_time))

    return G


def gdf_from_place(query, gdf_name=None, which_result=1, buffer_dist=None):
    # if no gdf_name is passed, just use the query
    assert (isinstance(query, dict) or isinstance(query, str)), 'query must be a dict or a string'
    if (gdf_name is None) and isinstance(query, dict):
        gdf_name = ', '.join(list(query.values()))
    elif (gdf_name is None) and isinstance(query, str):
        gdf_name = query

    # get the data from OSM
    data = osm_polygon_download(query, limit=which_result)
    if len(data) >= which_result:

        # extract data elements from the JSON response
        result = data[which_result - 1]
        bbox_south, bbox_north, bbox_west, bbox_east = [float(x) for x in result['boundingbox']]
        geometry = result['geojson']
        place = result['display_name']
        features = [{'type': 'Feature',
                     'geometry': geometry,
                     'properties': {'place_name': place,
                                    'bbox_north': bbox_north,
                                    'bbox_south': bbox_south,
                                    'bbox_east': bbox_east,
                                    'bbox_west': bbox_west}}]

        # if we got an unexpected geometry type (like a point), log a warning
        if geometry['type'] not in ['Polygon', 'MultiPolygon']:
            log('OSM returned a {} as the geometry.'.format(geometry['type']), level=lg.WARNING)

        # create the GeoDataFrame, name it, and set its original CRS to default_crs
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf.gdf_name = gdf_name
        gdf.crs = settings.default_crs

        # if buffer_dist was passed in, project the geometry to UTM, buffer it
        # in meters, then project it back to lat-long
        if buffer_dist is not None:
            gdf_utm = project_gdf(gdf)
            gdf_utm['geometry'] = gdf_utm['geometry'].buffer(buffer_dist)
            gdf = project_gdf(gdf_utm, to_latlong=True)
            log('Buffered the GeoDataFrame "{}" to {} meters'.format(gdf.gdf_name, buffer_dist))

        # return the gdf
        log('Created GeoDataFrame with {} row for query "{}"'.format(len(gdf), query))
        return gdf
    else:
        # if there was no data returned (or fewer results than which_result
        # specified)
        log('OSM returned no results (or fewer than which_result) for query "{}"'.format(query), level=lg.WARNING)
        gdf = gpd.GeoDataFrame()
        gdf.gdf_name = gdf_name
        return gdf


def graph_from_place(query, network_type='all_private', truncate_by_edge=False,
                     which_result=1, buffer_dist=None, timeout=180,
                     max_query_area_size=50 * 1000 * 50 * 1000, infrastructure='way["highway"]'):
    # create a GeoDataFrame with the spatial boundaries of the place(s)
    if isinstance(query, str):
        # if it is a string (place name) or dict (structured place query), then
        # it is a single place
        gdf_place = gdf_from_place(query, which_result=which_result, buffer_dist=buffer_dist)
        name = query
    else:
        raise TypeError('query must be a string')

    # extract the geometry from the GeoDataFrame to use in API query
    polygon = gdf_place['geometry'].unary_union
    log('Constructed place geometry polygon(s) to query API')

    # create graph using this polygon(s) geometry
    G = graph_from_polygon(polygon, network_type=network_type, truncate_by_edge=truncate_by_edge,
                           name=name, timeout=timeout, max_query_area_size=max_query_area_size,
                           infrastructure=infrastructure)

    log('graph_from_place() returning graph with {:,} nodes and {:,} edges'.format(len(list(G.nodes())),
                                                                                   len(list(G.edges()))))
    return G


def graph_from_file(filename, bidirectional=False, retain_all=True, name='unnamed'):
    # transmogrify file of OSM XML data into JSON
    response_jsons = [overpass_json_from_file(filename)]

    # create graph using this response JSON
    G = create_graph(response_jsons, bidirectional=bidirectional, name=name)

    log('graph_from_file() returning graph with {:,} nodes and {:,} edges'.format(len(list(G.nodes())),
                                                                                  len(list(G.edges()))))
    return G


def download_buildings_n_pois(query, selector_buildings={"building": [""]}, selector_pois={"amenity": ["hospital", "fire_station"], "shop": ['supermarket']}):
    buildings = []
    pois = []
    counter = [0, 0]
    areaId = Nominatim().query(query).areaId()
    overpass = Overpass()
    start_time = time.time()
    for key in selector_buildings:
        s = '"'+str(key)+'"'
        for e in selector_buildings[key]:
            if e:
                s += '="'+str(e)+'"'
            s = [s]
            query = overpassQueryBuilder(area=areaId, elementType=['node'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                for tag in selector_pois:
                    if tag in el.tags():
                        for value in selector_pois[tag]:
                            if el.tags()[tag] == value:
                                continue
                buildings.append(dict(id=el.id(), y=el.lat(), x=el.lon(), weight=1))
                counter[0] += 1

            query = overpassQueryBuilder(area=areaId, elementType=['way'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                for tag in selector_pois:
                    if tag in el.tags():
                        for value in selector_pois[tag]:
                            if el.tags()[tag] == value:
                                continue
                try:
                    point = el.nodes()[0]
                    buildings.append(dict(id=el.id(), y=point.lat(), x=point.lon(), weight=1))
                    counter[0] += 1
                except (urllib.error.HTTPError, Exception):
                    continue

            query = overpassQueryBuilder(area=areaId, elementType=['rel'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                for tag in selector_pois:
                    if tag in el.tags():
                        for value in selector_pois[tag]:
                            if el.tags()[tag] == value:
                                continue
                try:
                    way = el.members()[0]
                    point = way.nodes()[0]
                    buildings.append(dict(id=el.id(), y=point.lat(), x=point.lon(), weight=1))
                    counter[0] += 1
                except (urllib.error.HTTPError, Exception):
                    continue
            s = '"' + str(key) + '"'

    for k in selector_pois:
        s = '"'+str(k)+'"'
        for i in selector_pois[k]:
            if i:
                s += '="'+str(i)+'"'
            s = [s]
            query = overpassQueryBuilder(area=areaId, elementType=['node'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                pois.append(dict(id=el.id(), y=el.lat(), x=el.lon(), weight=2-random.random()))
                counter[1] += 1
            query = overpassQueryBuilder(area=areaId, elementType=['way'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                try:
                    point = el.nodes()[0]
                    pois.append(dict(id=el.id(), y=point.lat(), x=point.lon(), weight=2-random.random()))
                    counter[1] += 1
                except (urllib.error.HTTPError, Exception):
                    continue
            query = overpassQueryBuilder(area=areaId, elementType=['rel'], selector=s, out='meta')
            result = overpass.query(query)
            for el in result.elements():
                try:
                    way = el.members()[0]
                    point = way.nodes()[0]
                    pois.append(dict(id=el.id(), y=point.lat(), x=point.lon(), weight=2-random.random()))
                    counter[1] += 1
                except (urllib.error.HTTPError, Exception):
                    continue
            s = '"' + str(k) + '"'
    log('get_buildings_n_pois() returning {:,} buildings and {:,} places of interest in {:,.2f} seconds'.format(counter[0], counter[1], time.time() - start_time))
    return buildings, pois
