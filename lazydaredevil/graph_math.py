from heapq import heappush, heappop
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import time

from .utils import log
from .geo_utils import get_nearest_node, euclidean_dist_vec
from .settings import *

unreachable = 10000000000


def _weight(G, weight):
    return lambda u, v, d: d['length']


def path_deijcstra(G, source, targets):
    if source is None:
        raise ValueError('source must not be empty')
    if targets is None:
        raise ValueError('target must not be empty')
    r = {}
    if source in targets:
        r[source] = [0, [source]]
    if source not in G:
       raise ValueError("Source {} not in G".format(source))
    for el in targets:
        if el not in G:
            raise ValueError("Target {} not in G".format(targets))
    start_time = time.time()
    dist = {}
    path = {}
    seen = {}
    path[source] = [source]
    seen[source] = 0
    fringe = []
    push = heappush
    pop = heappop
    push(fringe, (0, source))
    weight = _weight(G, 'weight')
    while fringe:
        (d, v) = pop(fringe)
        if v in dist:
            continue
        dist[v] = d
        from_v = list(G.edges(nbunch=[v], data=True))
        for _, u, e in from_v:
            cost = weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + cost
            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, u))
                path[u] = path[v] + [u]
    for el in targets:
        if el not in path:
            r[el] = [unreachable, [el]]
        else:
            r[el] = [dist[el], path[el]]
    # log('deijcstra in {:,.2f} seconds'.format(time.time() - start_time))
    return r


def paths_n_lengths(G, targets, sources):
    t_len = len(targets)
    s_len = len(sources)
    N = t_len
    M = s_len
    distances_from_t_to_s = np.zeros((t_len, s_len))
    distances_from_s_to_t = np.zeros((s_len, t_len))
    paths_from_t_to_s = []
    paths_from_s_to_t = []
    for i in np.arange(M):
        paths_from_s_to_t.append([])
        for j in np.arange(N):
            paths_from_s_to_t[i].append([])
    for i in np.arange(N):
        paths_from_t_to_s.append([])
        for j in np.arange(M):
            paths_from_t_to_s[i].append([])
    start_time = time.time()
    for i in np.arange(M):
        t = []
        for el in targets:
            t.append(el['id_graph'])
        tmp = path_deijcstra(G, sources[i]['id_graph'], t)
        for key in tmp:
            j = 0
            for el in targets:
                if el['id_graph'] == key:
                    j = targets.index(el)
            distances_from_s_to_t[i][j] = tmp[key][0]
            paths_from_s_to_t[i][j] = tmp[key][1].copy()
    for i in np.arange(N):
        t = []
        for el in sources:
            t.append(el['id_graph'])
        tmp = path_deijcstra(G, targets[i]['id_graph'], t)
        for key in tmp:
            j = 0
            for el in sources:
                if el['id_graph'] == key:
                    j = sources.index(el)
            distances_from_t_to_s[i][j] = tmp[key][0]
            paths_from_t_to_s[i][j] = tmp[key][1].copy()
    log('Calculated distances for all targets and sources in {:,.2f} seconds'.format(time.time() - start_time))
    return distances_from_t_to_s, distances_from_s_to_t, paths_from_t_to_s, paths_from_s_to_t


def paths_n_lengths_from_s_to_t(G, targets, sources):
    t_len = len(targets)
    s_len = len(sources)
    N = t_len
    M = s_len
    distances_from_s_to_t = np.zeros((s_len, t_len))
    paths_from_s_to_t = []
    for i in np.arange(M):
        paths_from_s_to_t.append([])
    start_time = time.time()
    for i in np.arange(M):
        t = []
        for el in targets:
            t.append(el['id_graph'])
        tmp = path_deijcstra(G, sources[i]['id_graph'], t)
        for key in tmp:
            j = 0
            for el in targets:
                if el['id_graph'] == key:
                    j = targets.index(el)
            distances_from_s_to_t[i][j] = tmp[key][0]
            paths_from_s_to_t[i].append(tmp[key][1].copy())
    log('Calculated distances for all targets and sources in {:,.2f} seconds'.format(time.time() - start_time))
    return distances_from_s_to_t, paths_from_s_to_t


def test_task(G, targets, sources):
    t_len = len(targets)
    s_len = len(sources)
    N = t_len
    M = s_len
    distances_from_s_to_t = np.zeros((s_len, t_len))
    start_time = time.time()
    for i in np.arange(M):
        print(i)
        t = []
        for el in targets:
            t.append(el['id_graph'])
        tmp = path_deijcstra(G, sources[i]['id_graph'], t)
        for key in tmp:
            j = 0
            for el in targets:
                if el['id_graph'] == key:
                    j = targets.index(el)
            distances_from_s_to_t[i][j] = tmp[key][0]
    log('Calculated distances for all targets and sources in {:,.2f} seconds'.format(time.time() - start_time))
    return distances_from_s_to_t


def task_1_1_a(buildings, pois, distances_from_b_to_p, distances_from_p_to_b):
    b_len = len(buildings)
    p_len = len(pois)
    p_nearest_there = []
    p_nearest_here = []
    p_nearest_there_n_here = []
    for i in np.arange(p_len):
        p_nearest_there.append((unreachable, 0))
        p_nearest_here.append((unreachable, 0))
        p_nearest_there_n_here.append((unreachable, 0))
    for i in np.arange(p_len):
        for j in np.arange(b_len):
            if distances_from_p_to_b[i][j] < p_nearest_there[i][0]:
                p_nearest_there[i] = (distances_from_p_to_b[i][j], buildings[j]['id'])
            if distances_from_b_to_p[j][i] < p_nearest_here[i][0]:
                p_nearest_here[i] = (distances_from_b_to_p[j][i], buildings[j]['id'])
            if distances_from_p_to_b[i][j]+distances_from_b_to_p[j][i] < p_nearest_there_n_here[i][0]:
                p_nearest_there_n_here[i] = (distances_from_p_to_b[i][j]+distances_from_b_to_p[j][i], buildings[j]['id'])
    log('Found nearest buildings for all pois')
    return p_nearest_there, p_nearest_here, p_nearest_there_n_here


def task_1_1_b(buildings, pois, distances_from_b_to_p, distances_from_p_to_b, max_distance):
    b_len = len(buildings)
    p_len = len(pois)
    p_nearest_there = []
    p_nearest_here = []
    p_nearest_there_n_here = []
    if max_distance >= 1000000:
        log('Too big distance')
    else:
        for i in np.arange(p_len):
            p_nearest_there.append([])
            p_nearest_here.append([])
            p_nearest_there_n_here.append([])
        for i in np.arange(p_len):
            for j in np.arange(b_len):
                if distances_from_p_to_b[i][j] <= max_distance:
                    p_nearest_there[i].append((distances_from_p_to_b[i][j], buildings[j]['id']))
                if distances_from_b_to_p[j][i] <= max_distance:
                    p_nearest_here[i].append((distances_from_b_to_p[j][i], buildings[j]['id']))
                if distances_from_p_to_b[i][j]+distances_from_b_to_p[j][i] <= max_distance:
                    p_nearest_there_n_here[i].append((distances_from_p_to_b[i][j]+distances_from_b_to_p[j][i], buildings[j]['id']))
        log('Found buildings that not further than {:,} for all pois'.format(max_distance))
    return p_nearest_there, p_nearest_here, p_nearest_there_n_here


def task_1_2(buildings, pois, distances_from_p_to_b):
    b_len = len(buildings)
    p_len = len(pois)
    p_farthest = []
    for i in np.arange(p_len):
        p_farthest.append((0, 0))
    for i in np.arange(p_len):
        for j in np.arange(b_len):
            if (distances_from_p_to_b[i][j] > p_farthest[i][0]) and (distances_from_p_to_b[i][j] < unreachable):
                p_farthest[i] = (distances_from_p_to_b[i][j], buildings[j]['id'])
    p_with_min_farthest_b = (0, (unreachable, 0))
    for i in np.arange(p_len):
        if (p_farthest[i][0] < p_with_min_farthest_b[1][0]) and (p_farthest[i][0] > 0):
            p_with_min_farthest_b = (pois[i]['id'], p_farthest[i])
    log('Found poi with min farthest building')
    return p_with_min_farthest_b


def get_p_sum(targets, distances_form_s_to_t):
    p_sum_ways = 0
    b_len = len(targets)
    for j in np.arange(b_len):
        if distances_form_s_to_t[j] < unreachable:
            p_sum_ways += distances_form_s_to_t[j]
    return p_sum_ways


def task_1_3(buildings, pois, distances_from_p_to_b):
    b_len = len(buildings)
    p_len = len(pois)
    p_sum_ways = np.zeros(p_len)
    for i in np.arange(p_len):
        p_sum_ways[i] = get_p_sum(buildings, distances_from_p_to_b[i])
    p_min_sum = (0, unreachable)
    for i in np.arange(p_len):
        if (p_sum_ways[i] < p_min_sum[1]) and (p_sum_ways[i] < unreachable) and (p_sum_ways[i] > 0):
            p_min_sum = (pois[i]['id'], p_sum_ways[i])
    log('Found poi with minimal sum of ways')
    return p_min_sum


def common_members(a, b):
    a_set = set(a)
    b_set = set(b)
    common = []
    if a_set & b_set:
        common = a_set & b_set
    return common


def sort_dict_by_items(d):
    return sorted(d.items(), key=lambda kv: (kv[1], kv[0]))


def build_tree(source, targets, ways):
    tree = {source['id_graph']: [[source['id_graph']]]}
    b_len = len(targets)
    for j in np.arange(b_len):
        path = ways[j]
        keys = list(tree.keys()).copy()
        for key in keys:
            paths = list(tree[key]).copy()
            for x in paths:
                common = common_members(path, x)
                if len(common) > 1:
                    indexes_old = {}
                    indexes_new = {}
                    for el in common:
                        indexes_old[el] = x.index(el)
                        indexes_new[el] = path.index(el)
                    indexes_old = sort_dict_by_items(indexes_old)
                    indexes_new = sort_dict_by_items(indexes_new)
                    indexes_new_start = indexes_new[0][1]
                    indexes_new_prev = indexes_new[0][1]
                    indexes_old_start = indexes_old[0][1]
                    indexes_old_end = indexes_old[0][1]
                    for y in indexes_new[1:]:
                        indexes_new_now = y[1]
                        if indexes_new_now == indexes_new_prev + 1:
                            indexes_new_prev = indexes_new_now
                            indexes_old_end += 1
                        else:
                            break
                    indexes_new_end = indexes_new_prev
                    tmp = tree[key][tree[key].index(x)][indexes_old_end:]
                    if len(tmp) > 1:
                        if tmp[0] in tree:
                            tree[tmp[0]].append(tmp)
                        else:
                            tree[tmp[0]] = [tmp]
                    tree[key][tree[key].index(x)] = tree[key][tree[key].index(x)][indexes_old_start:indexes_old_end + 1]
                    path = path[indexes_new_end:]
        if path:
            if path[0] in tree:
                tree[path[0]].append(path)
            else:
                tree[path[0]] = [path]
    return tree


def build_trees_for_pois(buildings, pois, paths_from_p_to_b):
    trees = {}
    b_len = len(buildings)
    p_len = len(pois)
    for i in np.arange(p_len):
        trees[pois[i]['id']] = build_tree(pois[i], buildings, paths_from_p_to_b[i])
    return trees


def tree_weight(G, tree):
    p_tree_weight = 0
    for key in tree.keys():
        for path in tree[key]:
            p = len(path)
            u = path[0]
            i = 1
            while i < p:
                v = path[i]
                p_tree_weight += G.edges[u, v, 0]['length']
                u = v
                i += 1
    return p_tree_weight


def task_1_4(G, trees):
    p_min_tree_weight = (0, unreachable)
    for poi in trees.keys():
        p_tree_weight = tree_weight(G, trees[poi])
        if (p_tree_weight < p_min_tree_weight[1]) and (p_tree_weight > 0):
            p_min_tree_weight = (poi, p_tree_weight)
    return p_min_tree_weight


def task_2_2(buildings, pois, index_of_poi, numbers_of_clusters=[2, 3, 5]):
    p_len = len(pois)
    if index_of_poi >= p_len:
        log('Choosen poi doesnt exist')
        return []
    b_len = len(buildings)
    matrix = np.zeros((b_len, b_len))
    for i in np.arange(b_len):
        for j in np.arange(i, b_len):
            if i != j:
                tmp = euclidean_dist_vec(y1=buildings[i]['y'], x1=buildings[i]['x'],
                                         y2=buildings[j]['y'], x2=buildings[j]['x'])
                matrix[i][j] = tmp
                matrix[j][i] = tmp
    n = len(numbers_of_clusters)
    if n > 0:
        numbers_of_clusters.sort()
        tmp = []
        i = 0
        while i < n and numbers_of_clusters[i] < b_len:
            tmp.append(numbers_of_clusters[i])
            i += 1
        if len(tmp) < 1:
            tmp = [2]
        numbers_of_clusters = tmp
    else:
        numbers_of_clusters = [2]
    clusters = {}
    cluster = []
    cluster1 = []
    result = []
    for i in np.arange(b_len):
        cluster.append([i])
        cluster1.append([i])
    m = matrix.copy()

    while len(cluster) > 1:
        min_el = ([0, 0], unreachable)
        x = len(cluster)
        for i in np.arange(x):
            for j in np.arange(i, x):
                if i != j and m[i][j] < min_el[1]:
                    min_el = ([i, j], m[i][j])

        min_el[0].sort()
        if cluster1.index(cluster[min_el[0][0]]) != cluster1.index(cluster[min_el[0][1]]):
            result.append([cluster1.index(cluster[min_el[0][0]]), cluster1.index(cluster[min_el[0][1]]), min_el[1], \
                           len(cluster1[cluster1.index(cluster[min_el[0][0]])]) + len(cluster1[cluster1.index(cluster[min_el[0][1]])])])
            cluster[min_el[0][0]] += cluster[min_el[0][1]]
            cluster1.append(cluster[min_el[0][0]].copy())
            cluster.pop(min_el[0][1])

        for i in np.arange(x):
            for j in np.arange(x):
                if i != j:
                    if i == min_el[0][0]:
                        m[i][j] = max(m[i][j], m[min_el[0][1]][j])
                    if j == min_el[0][0]:
                        m[i][j] = max(m[i][j], m[i][min_el[0][1]])
        x = len(cluster)
        m_tmp = np.zeros((x, x))
        for i in np.arange(x):
            for j in np.arange(x):
                if i < min_el[0][1]:
                    if j < min_el[0][1]:
                        m_tmp[i][j] = m[i][j]
                    else:
                        m_tmp[i][j] = m[i][j+1]
                else:
                    if j < min_el[0][1]:
                        m_tmp[i][j] = m[i+1][j]
                    else:
                        m_tmp[i][j] = m[i+1][j+1]
        m = m_tmp
        if x in numbers_of_clusters:
            clusters[x] = cluster.copy()
            log('Made {:} clusters'.format(x))
    fig = plt.figure(figsize=(50, 20))
    dn = dendrogram(np.array(result))
    plt.savefig(task_folder +'/clucterisation_'+str(pois[index_of_poi]['id'])+'.png', format='png')
    log('Clusterisation was finished, look at the dendrogram')

    for key in clusters:
        tmp = []
        for cl in clusters[key]:
            tmp1 = []
            for el in cl:
                tmp1.append(buildings[el]['id'])
            tmp.append(tmp1)
        clusters[key] = tmp

    return clusters


def centre_of_masses(masses):
    return sum(masses)/len(masses)


def task_2_3_a(buildings, clusters):
    centroids = {}
    for key in clusters:
        c = []
        for i in np.arange(len(clusters[key])):
            x_s = []
            y_s = []
            for j in clusters[key][i]:
                a = next(k for k in buildings if k['id'] == j)
                x_s.append(a['x'])
                y_s.append(a['y'])
            c.append([centre_of_masses(x_s), centre_of_masses(y_s)])
        centroids[key] = c
    return centroids


def centriods_to_cpois(G, centroids):
    cpois_for_clusters = {}
    for key in centroids:
        cpois = []
        i = 1
        for el in centroids[key]:
            cpois.append({'id': i*10 + int(key), 'x': el[0], 'y': el[1], 'id_graph': get_nearest_node(G, [el[1], el[0]]), 'weight': 1})
            i += 1
        cpois_for_clusters[key] = cpois
    return cpois_for_clusters


def task_2_3_bc(G, buildings, pois, cpois_for_clusters, index_of_poi):
    p_len = len(pois)
    if index_of_poi >= p_len:
        log('Choosen poi doesnt exist')
        return {}
    trees_for_clusters = {}
    distances_from_cp_to_b_for_clusters = {}
    distances_from_p_to_cp_for_clusters = {}
    poi = pois[index_of_poi]
    for key in cpois_for_clusters:
        cpois = cpois_for_clusters[key]
        distances_from_b_to_cp, distances_from_cp_to_b, paths_from_b_to_cp, paths_from_cp_to_b = paths_n_lengths(G, buildings, cpois)
        distances_from_cp_to_b_for_clusters[key] = distances_from_cp_to_b
        cpois_trees = {}
        for cpoi in cpois:
            cpois_trees[cpoi['id']] = build_tree(cpoi, buildings, paths_from_cp_to_b[cpois_for_clusters[key].index(cpoi)])
        distances_from_cp_to_p, distances_from_p_to_cp, paths_from_cp_to_p, paths_from_p_to_cp = paths_n_lengths(G, cpois, [poi])
        distances_from_p_to_cp_for_clusters[key] = distances_from_p_to_cp
        poi_tree = build_tree(poi, cpois, paths_from_p_to_cp[0])
        for k in cpois_trees:
            for kk in cpois_trees[k]:
                poi_tree[kk] = cpois_trees[k][kk]
        trees_for_clusters[key] = poi_tree
    log('Built trees for all clusters for {:} poi'.format(poi['id']))
    return trees_for_clusters, distances_from_cp_to_b_for_clusters, distances_from_p_to_cp_for_clusters


def task_2_3_d(G, buildings, pois, cpois_for_clusters, index_of_poi, trees_for_clusters,
               distances_from_cp_to_b_for_clusters, distances_from_p_to_cp_for_clusters):
    p_len = len(pois)
    if index_of_poi > p_len:
        log('Choosen poi doesnt exist')
        return {}
    lenghts_for_clusters = {}
    for key in cpois_for_clusters:
        tmp = {'len_tree': 0, 'sum_ways': 0}
        for el in cpois_for_clusters[key]:
            tmp['len_tree'] += tree_weight(G, trees_for_clusters[key])
            tmp['sum_ways'] += get_p_sum(buildings,
                                         distances_from_cp_to_b_for_clusters[key][cpois_for_clusters[key].index(el)])
            if distances_from_p_to_cp_for_clusters[key][0][cpois_for_clusters[key].index(el)] < unreachable:
                tmp['sum_ways'] += distances_from_p_to_cp_for_clusters[key][0][cpois_for_clusters[key].index(el)]
        lenghts_for_clusters[key] = tmp
    return lenghts_for_clusters


def clusters_in_graph(G, clusters, buildings):
    c = {}
    for key in clusters:
        clust = []
        for cl in clusters[key]:
            cluster = []
            for el in cl:
                point = 0
                for e in buildings:
                    if e['id'] == el:
                        point = e['id_graph']
                cluster.append(point)
            clust.append(cluster)
        c[key] = clust
    return c
