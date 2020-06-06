import networkx as nx
import time
import random

from .utils import log
from .geo_utils import get_nearest_node


def choosing_buildings_n_pois(all_buildings, all_pois, N=100, M=10, method='random'):
    buildings = []
    pois = []
    if N > 100 or N <= 0 or M > 10 or M <= 0:
        log("N and M should be more than 0 and less than 100 and 10")
    else:
        if method == 'random':
            start_time = time.time()
            all_b = all_buildings.copy()
            all_p = all_pois.copy()
            b_len = len(all_b)
            p_len = len(all_p)
            if N >= b_len:
                log("N bigger than number of buildings, all buildings will be returned")
                buildings = all_b
            else:
                while N > 0:
                    next_choice = random.choice(all_b)
                    buildings.append(next_choice)
                    all_b.remove(next_choice)
                    N -= 1
            if M >= p_len:
                log("M bigger than number of places of interest, all pois will be returned")
                pois = all_p
            else:
                while M > 0:
                    next_choice = random.choice(all_p)
                    pois.append(next_choice)
                    all_p.remove(next_choice)
                    M -= 1
            log('choosing_buildings_n_pois() returning {:,} buildings, {:,} pois in {:,.2f} seconds'.format(len(buildings), len(pois), time.time() - start_time))
        else:
            log("Sorry, other methods isn't exist now")
    return buildings, pois


def find_nearest_nodes_in_graph(G, buildings, pois):
    new_buildings = buildings
    new_pois = pois
    for el in new_buildings:
        el['id_graph'] = get_nearest_node(G, (el['y'], el['x']), method='euclidean')
    for el in new_pois:
        el['id_graph'] = get_nearest_node(G, (el['y'], el['x']), method='euclidean')
    return new_buildings, new_pois