from .getdata import *
from .utils import *
from .download import *
from .projection import *
from .errors import *
from .projection import *
from .settings import *
from .geo_utils import *
from .osm_content_handler import *
from .save_load import *
from .plot import *
from .graph_math import *
from .working_with_data import *

import numpy as np
import datetime
import networkx as nx
import random
import pandas

"""
LazyDareDevil's graph project
version of osmnx made only for education
"""

__version__ = '0.1'


def main():
    city = 'Samara, Russia'
    G = None
    all_buildings = []
    buildings = []
    all_pois = []
    pois = []
    distances_from_b_to_p, distances_from_p_to_b, paths_from_b_to_p, paths_from_p_to_b = [], [], [], []
    trees = {}
    trees_for_clusters = {}
    log('Welcome!')
    while True:
        log('You want to use information about {}? y/n'.format(city))
        input_str = str(input())
        while input_str != 'y' and input_str != 'n' and input_str != 'quit()':
            log('Please, answer the question')
            input_str = str(input())
        if input_str == 'quit()':
            break
        if input_str == 'n':
            input_str = '0'
            while int(input_str) != 1 or int(input_str) != 2 or input_str != 'quit()':
                log('What you want to do?\n'
                    '    1 - choose another city and download its data\n    2 - do task to check if app is correct')
                input_str = input()
                if input_str == 'quit()':
                    break
                if int(input_str) == 1:
                    log('Sorry, that part isnt complete yet')
                    break
                if int(input_str) == 2:
                    log('Input name of test file:')
                    while G is None:
                        try:
                            input_str = str(input())
                            f = open(input_str, 'r')
                            num_lines = sum(1 for line in f) - 1
                            df = pandas.read_csv(input_str, names=np.arange(num_lines), header=0)
                            G = nx.DiGraph(np.array(df.values))
                            log('Got data from file')
                            for ed in G.edges(data=True):
                                ed[2]['length'] = ed[2]['weight']
                            buildings = []
                            pois = []
                            for i in np.arange(num_lines):
                                buildings.append({'id': i, 'id_graph': i, 'weight': 1})
                                pois.append({'id': i, 'id_graph': i, 'weight': 2 - random.random()})
                            log('Calculating started...')
                            distances_from_b_to_p, distances_from_p_to_b, paths_from_b_to_p, paths_from_p_to_b = paths_n_lengths(
                                G, buildings, pois)
                            pandas.DataFrame(distances_from_p_to_b).to_csv('task_results/examples_result.csv')
                            log('Saved result distances')
                            input_str = '0'
                            break
                        except Exception:
                            log('Some errors occurred, sorry...')
                            input_str = '0'
                            break

        while input_str != 'quit()':
            show_results_into_console = True
            log('Download data about {} from file? y/n'.format(city))
            input_str = str(input())
            if input_str == 'quit()':
                break
            while input_str != 'y' and input_str != 'n':
                log('Please, answer the question')
                input_str = str(input())
            if input_str == 'n':
                try:
                    log('Downloading and parsing network graph started...\nIt may be long...')
                    G = graph_from_place(city, network_type='all')
                    log('Downloading and parsing nodes started...\nIt may be long...')
                    all_buildings, all_pois = download_buildings_n_pois(city)
                except Exception:
                    log('Some errors occurred, sorry...')
                    break
            else:
                input_str = ''
                while G is None:
                    log('Input network data file name:')
                    input_str = str(input())
                    try:
                        log('Getting graph data started...')
                        G = graph_from_file(input_str)
                        log('Plot graph? y/n')
                        input_str = str(input())
                        while input_str != 'y' and input_str != 'n':
                            log('Please, answer the question')
                            input_str = str(input())
                        if input_str == 'y':
                            plot_graph(G, filename=city)
                    except Exception:
                        log('Cannot load data from file "{}"'.format(input_str))
                b = ''
                p = ''
                while len(all_buildings) == 0 or len(all_pois) == 0:
                    log('Input buildings data file name:')
                    b = str(input())
                    log('Input pois data file name:')
                    p = str(input())
                    log('Getting buildings and pois data started...')
                    all_buildings, all_pois = get_buildings_n_pois(filename_buildings=b, filename_pois=p)

            log('Log data into console? y/n'.format(city))
            input_str = str(input())
            log_file = open('task_results/tasks.txt', 'a')
            while input_str != 'y' and input_str != 'n':
                log('Please, answer the question')
                input_str = str(input())
            if input_str == 'n':
                show_results_into_console = False
                log_file.write(str(datetime.datetime.today()) + '\n')
            else:
                log_file.close()
            while True:
                log('Whats next?\n    0 - choose buildings and pois\n'
                    '    1 - task 1\n    2 - task 2\n    3 - save data into csv\n 10 - quit')
                input_data = int(input())
                if input_data == 0:
                    log('Input number of buildings less than 100')
                    b = int(input())
                    log('Input number of pois less than 10')
                    p = int(input())
                    buildings, pois = choosing_buildings_n_pois(all_buildings, all_pois, N=b, M=p)
                    log('Getting their nearest nodes in graph...')
                    buildings, pois = find_nearest_nodes_in_graph(G, buildings, pois)
                    log('Getting ways and lengths between pois and buildings...')
                    distances_from_b_to_p, distances_from_p_to_b, paths_from_b_to_p, paths_from_p_to_b = \
                        paths_n_lengths(G, buildings, pois)

                if input_data == 1:
                    if len(distances_from_b_to_p) == 0 or len(distances_from_p_to_b) == 0:
                        log('Please, choose buildings and pois first')
                        continue
                    if not show_results_into_console:
                        log_file.write('Task 1\n')
                    log('Task 1a')
                    if not show_results_into_console:
                        log_file.write('Task 1a\n')
                    p_nearest_there, p_nearest_here, p_nearest_there_n_here = task_1_1_a(buildings, pois,
                                                                                         distances_from_b_to_p,
                                                                                         distances_from_p_to_b)
                    for poi in pois:
                        if show_results_into_console:
                            log(' for poi {}'.format(poi['id']))
                            index = pois.index(poi)
                            if p_nearest_there[index][1] != 0:
                                log('   nearest node {} in distance {}'.format(p_nearest_there[index][1],
                                                                               p_nearest_there[index][0]))
                            else:
                                log('   no paths to buildings')
                            if p_nearest_here[index][1] != 0:
                                log('   poi nearest from {} node in distance {}'.format(p_nearest_here[index][1],
                                                                                        p_nearest_here[index][0]))
                            else:
                                log('   no paths from buildings')
                            if p_nearest_there_n_here[index][1] != 0:
                                log('   nearest for {} node in distance {} for there and here way'.format(
                                    p_nearest_there_n_here[index][1], p_nearest_there_n_here[index][0]))
                            else:
                                log('   no paths to or from buildings')
                        else:
                            log_file.write(' for poi {}'.format(poi['id']) + '\n')
                            index = pois.index(poi)
                            if p_nearest_there[index][1] != 0:
                                log_file.write('   nearest node {} in distance {}'.format(p_nearest_there[index][1],
                                                                                          p_nearest_there[index][
                                                                                              0]) + '\n')
                            else:
                                log_file.write('   no paths to buildings' + '\n')
                            if p_nearest_here[index][1] != 0:
                                log_file.write(
                                    '   poi nearest from {} node in distance {}'.format(p_nearest_here[index][1],
                                                                                        p_nearest_here[index][
                                                                                            0]) + '\n')
                            else:
                                log_file.write('   no paths from buildings' + '\n')
                            if p_nearest_there_n_here[index][1] != 0:
                                log_file.write('   nearest for {} node in distance {} for there and here way'.format(
                                    p_nearest_there_n_here[index][1], p_nearest_there_n_here[index][0]) + '\n')
                            else:
                                log_file.write('   no paths to or from buildings' + '\n')

                    log('Task 1b. Input max distance:')
                    distance = float(input())
                    if not show_results_into_console:
                        log_file.write('Task 1b (less than {})\n'.format(distance))
                    p_nearest_there1, p_nearest_here1, p_nearest_there_n_here1 = task_1_1_b(buildings, pois,
                                                                                            distances_from_b_to_p,
                                                                                            distances_from_p_to_b,
                                                                                            distance)
                    for poi in pois:
                        if show_results_into_console:
                            log(' for poi {}'.format(poi['id']))
                            index = pois.index(poi)
                            p = p_nearest_there1[index]
                            i = 0
                            for el in p:
                                log('   node {} in distance {}'.format(el[1], el[0]))
                                i += 1
                            if i == 0:
                                log('   no paths to buildings less than {}'.format(distance))
                            p = p_nearest_here1[index]
                            i = 0
                            for el in p:
                                log('   to poi from {} node in distance {}'.format(el[1], el[0]))
                                i += 1
                            if i == 0:
                                log('   no paths from buildings less than {}'.format(distance))
                            p = p_nearest_there_n_here1[index]
                            i = 0
                            for el in p:
                                log('   for {} node in distance {} for there and here way'.format(el[1], el[0]))
                                i += 1
                            if i == 0:
                                log('   no paths to plus from buildings less than {}'.format(distance))
                        else:
                            log_file.write(' for poi {}'.format(poi['id']) + '\n')
                            index = pois.index(poi)
                            p = p_nearest_there1[index]
                            i = 0
                            for el in p:
                                log_file.write('   node {} in distance {}'.format(el[1], el[0]) + '\n')
                                i += 1
                            if i == 0:
                                log_file.write('   no paths to buildings less than {}'.format(distance) + '\n')
                            p = p_nearest_here1[index]
                            i = 0
                            for el in p:
                                log_file.write('   to poi from {} node in distance {}'.format(el[1], el[0]) + '\n')
                                i += 1
                            if i == 0:
                                log_file.write('   no paths from buildings less than {}'.format(distance) + '\n')
                            p = p_nearest_there_n_here1[index]
                            i = 0
                            for el in p:
                                log_file.write(
                                    '   for {} node in distance {} for there and here way'.format(el[1], el[0]) + '\n')
                                i += 1
                            if i == 0:
                                log_file.write(
                                    '   no paths to plus from buildings less than {}'.format(distance) + '\n')

                    log('Task 2')
                    if not show_results_into_console:
                        log_file.write('Task 2\n')
                    p_with_min_farthest_b = task_1_2(buildings, pois, distances_from_p_to_b)
                    if p_with_min_farthest_b[1][1] != 0:
                        if show_results_into_console:
                            log('poi {} having min way length {} to node {}'.format(p_with_min_farthest_b[0],
                                                                                    p_with_min_farthest_b[1][0],
                                                                                    p_with_min_farthest_b[1][1]))
                        else:
                            log_file.write('poi {} having min way length {} to node {}'.format(p_with_min_farthest_b[0],
                                                                                               p_with_min_farthest_b[1][
                                                                                                   0],
                                                                                               p_with_min_farthest_b[1][
                                                                                                   1]) + '\n')

                    else:
                        if show_results_into_console:
                            log('its seems like no ways')
                        else:
                            log_file.write('its seems like no ways\n')

                    log('Task 3')
                    if not show_results_into_console:
                        log_file.write('Task 3\n')
                    p_min_sum = task_1_3(buildings, pois, distances_from_p_to_b)
                    if p_min_sum[0] != 0:
                        if show_results_into_console:
                            log('poi {} having min sum of ways ({})'.format(p_min_sum[0], p_min_sum[1]))
                        else:
                            log_file.write(
                                'poi {} having min sum of ways ({})'.format(p_min_sum[0], p_min_sum[1]) + '\n')
                    else:
                        if show_results_into_console:
                            log('its seems like no ways')
                        else:
                            log_file.write('its seems like no ways\n')

                    log('Task 4')
                    if not show_results_into_console:
                        log_file.write('Task 4\n')
                    trees = build_trees_for_pois(buildings, pois, paths_from_p_to_b)
                    p_min_tree_weight = task_1_4(G, trees)
                    log('Found poi with min tree weight')
                    if p_min_tree_weight[0] != 0:
                        if show_results_into_console:
                            log('poi {} having min tree weight ({})'.format(p_min_tree_weight[0], p_min_tree_weight[1]))
                        else:
                            log_file.write('poi {} having min tree weight ({})'.format(p_min_tree_weight[0],
                                                                                       p_min_tree_weight[1]) + '\n')
                    else:
                        if show_results_into_console:
                            log('its seems like no ways')
                        else:
                            log_file.write('its seems like no ways\n')
                    log('Plotting trees... ')
                    plot_trees(G, trees)

                if input_data == 2:
                    if len(distances_from_b_to_p) == 0 or len(distances_from_p_to_b) == 0:
                        log('Please, choose buildings and pois first')
                        continue
                    if not show_results_into_console:
                        log_file.write('Task 2\n')
                    log('Choose poi by entering its index')
                    for i in np.arange(len(pois)):
                        log('{}: {}'.format(i, pois[i]['id']))
                    index_of_poi = input()
                    while type(index_of_poi) != int and (int(index_of_poi) >= len(pois)) and (int(index_of_poi) < 0):
                        log('Input real index of poi')
                        index_of_poi = int(input())

                    index_of_poi = int(index_of_poi)
                    log('Choosen poi: {}'.format(pois[index_of_poi]['id']))
                    log('Task 1')
                    if not show_results_into_console:
                        log_file.write('Task 1\n')
                    tree = trees[pois[index_of_poi]['id']]
                    tree_w = tree_weight(G, tree)
                    sum_w = get_p_sum(buildings, distances_from_p_to_b[index_of_poi])
                    log('Found tree weight and sum of lengths for poi {}'.format(pois[index_of_poi]['id']))
                    if show_results_into_console:
                        log('poi {} having {} tree weight and {} sum of ways'.format(pois[index_of_poi]['id'], tree_w,
                                                                                     sum_w))
                    else:
                        log_file.write(
                            'poi {} having {} tree weight and {} sum of ways'.format(pois[index_of_poi]['id'],
                                                                                     tree_w, sum_w) + '\n')

                    log('Task 2 and 3')
                    if not show_results_into_console:
                        log_file.write('Task 2 and 3\n')
                    log('Calculating clusters and centroids...')
                    clusters = task_2_2(buildings, pois, index_of_poi, numbers_of_clusters=[2, 3, 5])
                    log('Searching centriods...')
                    centroids = task_2_3_a(buildings, clusters)
                    cpois_for_clusters = centriods_to_cpois(G, centroids)
                    log('Building ways and trees for clusters...')
                    trees_for_clusters, distances_from_cp_to_b_for_clusters, distances_from_p_to_cp_for_clusters = \
                        task_2_3_bc(G, buildings, pois, cpois_for_clusters, index_of_poi)
                    log('Plotting trees for clusters...')
                    plot_trees_clusters(G, trees_for_clusters, clusters_in_graph(G, clusters, buildings),
                                        cpois_for_clusters, pois[index_of_poi])
                    log('Calculating sum lengths of ways and weights of trees...')
                    lengths_for_clusters = task_2_3_d(G, buildings, pois, cpois_for_clusters, index_of_poi,
                                                      trees_for_clusters,
                                                      distances_from_cp_to_b_for_clusters,
                                                      distances_from_p_to_cp_for_clusters)
                    if show_results_into_console:
                        log('for poi {}'.format(pois[index_of_poi]['id']))
                    else:
                        log_file.write('for poi {}'.format(pois[index_of_poi]['id']) + '\n')
                    for key in lengths_for_clusters:
                        if show_results_into_console:
                            log(' for {} clusters tree weight is {}, sum of ways is {}'.format(key,
                                                                                               lengths_for_clusters[
                                                                                                   key][
                                                                                                   'len_tree'],
                                                                                               lengths_for_clusters[
                                                                                                   key][
                                                                                                   'sum_ways']))
                        else:
                            log_file.write(' for {} clusters tree weight is {}, sum of ways is {}'.format(key,
                                                                                                          lengths_for_clusters[
                                                                                                              key][
                                                                                                              'len_tree'],
                                                                                                          lengths_for_clusters[
                                                                                                              key][
                                                                                                              'sum_ways']) + '\n')
                if input_data == 3:
                    log('Started saving data...')
                    # pandas.DataFrame(
                    #     nx.adjacency_matrix(G, nodelist=list(G.nodes), weight='length')).to_csv(
                    #     task_folder + '/adjacency_list.csv')
                    # nx.to_pandas_adjacency(G).to_csv(task_folder + '/adjacency_matrix.csv')
                    # log('Saved adjacency matrix')
                    pandas.DataFrame(nx.generate_adjlist(G)).to_csv(task_folder + '/adjacency_list.csv')
                    log('Saved adjacency list')
                    pandas.DataFrame(G.nodes(data=True)).to_csv(task_folder + '/nodes.csv')
                    log('Saved nodes data')
                    pandas.DataFrame(buildings).to_csv(task_folder + '/buildings.csv')
                    log('Saved buildings data')
                    pandas.DataFrame(pois).to_csv(task_folder + '/pois.csv')
                    log('Saved pois data')
                    pandas.DataFrame(paths_from_p_to_b).to_csv(task_folder + '/paths_from_pois_to_buildings.csv')
                    log('Saved paths from pois to buildings')
                    pandas.DataFrame(paths_from_b_to_p).to_csv(task_folder + '/paths_from_buildings_to_pois.csv')
                    log('Saved paths from buildings to pois')
                    pandas.DataFrame(trees).to_csv(task_folder + '/trees.csv')
                    log('Saved trees')
                    # for key in trees_for_clusters:
                    #     pandas.DataFrame(trees_for_clusters[key]).to_csv(task_folder + '/' + str(key) + '_clusters_tree.csv')
                    #     log('Saved tree for {} clusters'.format(key))
                if input_data == 10:
                    input_str = '0'
                    break
            if not show_results_into_console:
                log_file.close()
    return 'Thanks for using LazyDareDevil'
