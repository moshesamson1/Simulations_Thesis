import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt
from math import floor
import Entities


def print_graph(edges, i_o, figure_label=""):
    plt.figure()
    g = nx.DiGraph()
    g.add_edges_from(edges)
    positions = {}
    color_map = []
    for n in g.nodes():
        positions[n] = (n.row, n.col)
        if n == edges[0][0]:
            color_map.append('green')
        elif n == i_o:
            color_map.append('blue')
        else:
            color_map.append('red')
        # color_map.append('green' if n == edges[0][0] else 'red')

    f = plt.figure()
    nx.draw(G=g, pos=positions, node_size=20, font_size=8, node_color=color_map, ax=f.add_subplot(111))
    f.savefig(figure_label + "_max_path.png", bbox_inches='tight')
    plt.close('all')

    # plt.axis([-1, 33, -1, 33], 'on')
    # plt.grid(True)

    plt.show()


def create_graph(edge_list):
    graph = {}
    for e1, e2 in edge_list:
        graph.setdefault(e1, []).append(e2)
        graph.setdefault(e2, []).append(e1)
    return graph


def get_shallow_graph(original_edges):
    shallow_graph = {}
    for e1, e2 in original_edges:
        shallow_e1 = Entities.Slot(floor(e1.row / 2.0), floor(e1.col / 2.0))
        shallow_e2 = Entities.Slot(floor(e2.row / 2.0), floor(e2.col / 2.0))

        if (shallow_e1.row, shallow_e1.col) == (shallow_e2.row, shallow_e2.col):
            continue

        if shallow_e1 not in shallow_graph.keys() or shallow_e2 not in shallow_graph[shallow_e1]:
            shallow_graph.setdefault(shallow_e1, []).append(shallow_e2)

        if shallow_e2 not in shallow_graph.keys() or shallow_e1 not in shallow_graph[shallow_e2]:
            shallow_graph.setdefault(shallow_e2, []).append(shallow_e1)

    return shallow_graph


# Prim's
def mst(start, graph):
    closed = set()
    edges = []
    q = [(start, start)]
    while q:
        # randomize
        shuffle(q)

        v1, v2 = q.pop()
        if v2 in closed:
            continue
        closed.add(v2)
        edges.append((v1, v2))
        for v in graph[v2]:
            if v in graph:
                q.append((v2, v))
    del edges[0]
    assert len(edges) == len(graph)-1
    return edges


def create_covering_path(mst_edges_shallow_graph, initial_slot):
    covering_path = []
    origin_slot = initial_slot
    slot = origin_slot
    counter = 0
    while True:
        counter += 1
        if counter > 100000:
            print "ERROR! Probably hit infinite loop while creating the covering path"
            return

        covering_path.append(slot)
        shallow_slot = Entities.Slot(floor(slot.row / 2.0), floor(slot.col / 2.0))
        # find to where to go next, depend on the mst edges.
        # Check how much and which corners are in the mst group, then update slot accordingly

        has_downward_edge = (shallow_slot, shallow_slot.go_south()) in mst_edges_shallow_graph or \
                            (shallow_slot.go_south(), shallow_slot) in mst_edges_shallow_graph
        has_rightward_edge = (shallow_slot, shallow_slot.go_east()) in mst_edges_shallow_graph or \
                             (shallow_slot.go_east(), shallow_slot) in mst_edges_shallow_graph
        has_leftward_edge = (shallow_slot, shallow_slot.go_west()) in mst_edges_shallow_graph or \
                            (shallow_slot.go_west(), shallow_slot) in mst_edges_shallow_graph
        has_upward_edge = (shallow_slot, shallow_slot.go_north()) in mst_edges_shallow_graph or \
                          (shallow_slot.go_north(), shallow_slot) in mst_edges_shallow_graph

        bl_corner_in_mst = False
        br_corner_in_mst = False
        ul_corner_in_mst = False
        ur_corner_in_mst = False

        if slot.row % 2 == 0 and slot.col % 2 == 0:
            if has_downward_edge or has_rightward_edge:
                br_corner_in_mst = True
            if has_upward_edge:
                br_corner_in_mst = True
                ur_corner_in_mst = True
            if has_leftward_edge:
                bl_corner_in_mst = True
                br_corner_in_mst = True
        elif slot.row % 2 == 0 and slot.col % 2 != 0:
            if has_downward_edge or has_leftward_edge:
                bl_corner_in_mst = True
            if has_upward_edge:
                bl_corner_in_mst = True
                ul_corner_in_mst = True
            if has_rightward_edge:
                bl_corner_in_mst = True
                br_corner_in_mst = True
        elif slot.row % 2 != 0 and slot.col % 2 == 0:
            if has_rightward_edge or has_upward_edge:
                ur_corner_in_mst = True
            if has_downward_edge:
                br_corner_in_mst = True
                ur_corner_in_mst = True
            if has_leftward_edge:
                ur_corner_in_mst = True
                ul_corner_in_mst = True
        elif slot.row % 2 != 0 and slot.col % 2 != 0:
            if has_leftward_edge or has_upward_edge:
                ul_corner_in_mst = True
            if has_downward_edge:
                bl_corner_in_mst = True
                ul_corner_in_mst = True
            if has_rightward_edge:
                ul_corner_in_mst = True
                ur_corner_in_mst = True

        last_slot = covering_path[len(covering_path)-2]

        # check o see if only one vertex was in the mst
        if br_corner_in_mst and not (ur_corner_in_mst or bl_corner_in_mst or ul_corner_in_mst):
            if slot.go_south() == last_slot or slot == last_slot:
                slot = slot.go_east()
            elif slot.go_east() == last_slot:
                slot = slot.go_south()
        elif ur_corner_in_mst and not (br_corner_in_mst or bl_corner_in_mst or ul_corner_in_mst):
            if slot.go_east() == last_slot or slot == last_slot:
                slot = slot.go_north()
            elif slot.go_north() == last_slot:
                slot = slot.go_east()
        elif bl_corner_in_mst and not (ul_corner_in_mst or ur_corner_in_mst or br_corner_in_mst):
            if slot.go_west() == last_slot or slot == initial_slot:
                slot = slot.go_south()
            elif slot.go_south() == last_slot:
                slot = slot.go_west()
        elif ul_corner_in_mst and not (bl_corner_in_mst or br_corner_in_mst or ur_corner_in_mst):
            if slot.go_north() == last_slot or slot == initial_slot:
                slot = slot.go_west()
            elif slot.go_west() == last_slot:
                slot = slot.go_north()
        # check to see if exactly two vertices are in the mst
        elif bl_corner_in_mst and br_corner_in_mst and not (ul_corner_in_mst or ur_corner_in_mst):
            if slot.go_west() == last_slot or slot == initial_slot:
                slot = slot.go_east()
            elif slot.go_east() == last_slot:
                slot = slot.go_west()
            else:
                print "error 1"
        elif ul_corner_in_mst and ur_corner_in_mst and not (bl_corner_in_mst or br_corner_in_mst):
            if slot.go_east() == last_slot or slot == initial_slot:
                slot = slot.go_west()
            elif slot.go_west() == last_slot:
                slot = slot.go_east()
        elif br_corner_in_mst and ur_corner_in_mst and not (bl_corner_in_mst or ul_corner_in_mst):
            if slot.go_south() == last_slot or slot == initial_slot:
                slot = slot.go_north()
            elif slot.go_north() == last_slot:
                slot = slot.go_south()
        elif bl_corner_in_mst and ul_corner_in_mst and not (br_corner_in_mst or ur_corner_in_mst):
            if slot.go_north() == last_slot or slot == initial_slot:
                slot = slot.go_south()
            elif slot.go_south() == last_slot:
                slot = slot.go_north()
        # check for exactly 3 vertices
        elif br_corner_in_mst and bl_corner_in_mst and ul_corner_in_mst and not ur_corner_in_mst:
            if slot.go_north() == last_slot or slot == initial_slot:
                slot = slot.go_east()
            elif slot.go_east() == last_slot:
                slot = slot.go_north()
        elif bl_corner_in_mst and ul_corner_in_mst and ur_corner_in_mst and not br_corner_in_mst:
            if slot.go_south() == last_slot or slot == initial_slot:
                slot = slot.go_east()
            elif slot.go_east() == last_slot:
                slot = slot.go_south()
        elif ul_corner_in_mst and ur_corner_in_mst and br_corner_in_mst and not bl_corner_in_mst:
            if slot.go_west() == last_slot or slot == initial_slot:
                slot = slot.go_south()
            elif slot.go_south() == last_slot:
                slot = slot.go_west()
        elif ur_corner_in_mst and br_corner_in_mst and bl_corner_in_mst and not ul_corner_in_mst:
            if slot.go_north() == last_slot or slot == initial_slot:
                slot = slot.go_west()
            elif slot.go_west() == last_slot:
                slot = slot.go_north()
        else:
            print "error has occurred!"

        if slot == origin_slot:
            break

    return covering_path


def get_edges_for_full_graph(width, height):
    edges = []
    for row in xrange(height):
        for col in xrange(width):
            if row < height-1:
                edges.append((Entities.Slot(row, col), Entities.Slot(row, col).go_south()))
            if col < width-1:
                edges.append((Entities.Slot(row, col), Entities.Slot(row, col).go_east()))
    return edges


def get_random_coverage_strategy(size, i_r, i_o=None, print_mst=False, figure_label=""):
    edges = get_edges_for_full_graph(size, size)
    shallow_graph = get_shallow_graph(edges)
    shallow_init_pos = Entities.Slot(floor(i_r.row / 2.0), floor(i_r.col / 2.0))
    mst_edges_shallow_graph = mst(shallow_init_pos, shallow_graph)
    covering_path = create_covering_path(mst_edges_shallow_graph, i_r)
    if print_mst:
        covering_path_edges = []
        for i in xrange(0, len(covering_path)-1):
            covering_path_edges.append((covering_path[i], covering_path[i+1]))
        print_graph(covering_path_edges, i_o, figure_label=figure_label)
    return covering_path


def display_path(covering_path):
    covering_path_edges = []
    for i in xrange(0, len(covering_path) - 1):
        covering_path_edges.append((covering_path[i], covering_path[i + 1]))
    print_graph(covering_path_edges, covering_path_edges[0][0])


if __name__ == '__main__':
    get_random_coverage_strategy(32, Entities.Slot(16, 16), True)
