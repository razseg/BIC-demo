
import networkx as nx
import numpy as np
def DelayInit(g):
    for e in g.edges:
        nx.set_edge_attributes(g, {e: {'T': [0]}})
    return g

def delayCalc(g):
    root=len(g.nodes)-1
    l=list (nx.all_pairs_dijkstra_path_length(g))
    rootIndex=list(g.nodes).index(root)
    g = DelayInit(g)
    order=list(nx.bfs_tree(g,0))
    order.reverse()
    for node in order:
           nodeDealy(g,node)
    return np.max(np.where(np.array(g.edges[(root,0)]['T']) == 1))+1


def nodeDealy(g, node):
    n = [x for x in list(g.neighbors(node))]
    e_in = list(g.in_edges(node))
    if not n:  # leaf node
        if g.nodes[node]['color'] == 'blue':
            g.edges[e_in[0]]['T'][0] = 1
        else:
            tmp = []
            for i in range(0, g.nodes[node]['load']):
                tmp.append(1)
            g.edges[e_in[0]]['T'] = tmp
    else:
        e_out = list(g.out_edges(node))
        tmpD = [0]  # np.array([0]*len(g.edges[e_out[0]]))
        maxD = 0
        for e in e_out:
            if 1 in np.array(g.edges[e]['T']):
                if len(tmpD) < len(np.array(g.edges[e]['T'])):
                    c = np.array(g.edges[e]['T']).copy()
                    c[:len(tmpD)] += tmpD
                else:
                    c = tmpD.copy()
                    # print(np.array(g.edges[e]['T']))
                    c[:len(np.array(g.edges[e]['T']))] += np.array(g.edges[e]['T'])
                tmpD = c
                if maxD < np.max(np.where(np.array(g.edges[e]['T']) == 1)):
                    maxD = np.max(np.where(np.array(g.edges[e]['T']) == 1))

        if g.nodes[node]['color'] == 'blue':
            tmp = np.where(tmpD == 0)
            tmpD = [0] * (maxD + 2)
            tmpD[maxD + 1] = 1
            # g.edges[e_in[0]]['T']=[0]*len(g.edges[e_in[0]]['T'])
            # g.edges[e_in[0]]['T'][(np.min(tmp))+1]=1
            # tmpD=tmpD[0]

        else:
            tmpD = np.insert(tmpD, 0, 0)
            i = 0
            while max(tmpD) > 1:
                if tmpD[i] > 1:
                    if i + 1 >= len(tmpD):
                        tmpD = np.append(tmpD, tmpD[i] - 1)
                    else:
                        tmpD[i + 1] = tmpD[i] + tmpD[i + 1] - 1
                    tmpD[i] = 1
                i = i + 1
        if e_in:
            g.edges[e_in[0]]['T'] = list(tmpD)

