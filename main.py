import random

import path_alg as p_alg
import delay_clac as delay_c
import pi_tilde_alg as tilde
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from functions import *
import matplotlib.pyplot as plt
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

def SetBlue(g,Blist):
    for n in g.nodes:
        g.nodes[n]['color'] = 'red'
    for n in Blist:
        g.nodes[n]['color'] = 'blue'

def InitG(g):
    for n in g.nodes:
        att={n: {'color':'red','load':0}}
        nx.set_node_attributes(g,att)
    for e in g.edges:
        nx.set_edge_attributes(g,{e:{'T':[0]}})
    return g





deg=2
h=4
k=5
gr=nx.balanced_tree(deg, h, create_using=nx.DiGraph)
# gr = preferentialAttachment(30)
# lines = ["0 1","1 3","3 7","0 2","2 6","0 4","0 5"]
# gr = nx.parse_edgelist(lines, create_using=nx.DiGraph, nodetype=int)
gr.add_edge(len(gr.nodes), 0)
# load= [(7,2),(6,2),(4,1),(5,2)]

load =[(l, random.randint(1,10)) for l in leafList(gr)]
# load =[(l, 1) for l in leafList(gr)]
# loadp = list(np.random.permutation(power_law_distribution(len(leafList(gr)), 2, 1)))
leafL = leafList(gr)
# load=[(7, 8), (8, 2), (9, 10), (10, 6), (11, 6), (12, 4), (13, 1), (14, 7)]
# load = [(leafL[l], loadp[l]) for l in range(0, len(leafList(gr)))]
# load = [(8,20),(22,10),(3,10)]
# load = [(1, 1), (2, 3)]
# load =[(3, 1), (4, 3), (5, 3), (6, 1)]
# load =[(3, 2), (4, 10), (5, 13), (6, 3)]
# load = [(7, 10), (8, 5), (9, 1), (10, 5), (11, 1), (12,1 ), (13, 1), (14, 20)]
# load= [(7, 2), (8, 9), (9, 2), (10, 5), (11, 5), (12, 8), (13, 1), (14, 2)]
# load =[(3, 3), (4, 6), (5, 5), (6, 4)]
# gr = InitG(gr)
# SetLoad(gr, )
# SetBlue(gr, list(gr.nodes))
g = copy.deepcopy(gr)
g_soar, soar_col =soar_run(g, k, load)
plt.figure(1)
plotFig(g_soar,"SOAR")
# nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_soar))
print("delay SOAR:"+str(delay_c.delayCalc(g_soar)))
print("congestion SOAR:"+str(getCongestion(g_soar)))

# plt.figure(2)
# g = copy.deepcopy(gr)
# g_path = pi_run(g, k, load)
# print("delay path:"+str(delay_c.delayCalc(g_path)))
# print("congestion path:"+str(getCongestion(g_path)))
# # nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_path))
# plotFig(g_path,"Path")

# gr= nx.read_adjlist("tree.txt", nodetype=int, create_using=nx.DiGraph)
# load=[(3, 8), (6,7), (5, 3), (2, 2)]
# k=5
plt.close(3)
plt.figure(3)
g = copy.deepcopy(gr)
g_smc, smc_col =smc_run(g, k, load)
# nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_smc))
print("delay smc:"+str(delay_c.delayCalc(g_smc)))
print("congestion smc:"+str(getCongestion(g_smc)))
plotFig(g_smc,"SMC")

plt.close(4)
plt.figure(4)
g = copy.deepcopy(gr)
g_path_t = pi_tild_run(g, k, load)
print("delay path t:"+str(delay_c.delayCalc(g_path_t)))
print("congestion path t:"+str(getCongestion(g_path_t)))
print("utilization path t:"+str(getUtilization(g_path_t)))
# nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_path_t))
plotFig(g_path_t,"PathT")

# plt.close(5)
# plt.figure(5)
# g = copy.deepcopy(gr)
# g_path_tm,c = level_run(g, k, load)
# print("delay path t:"+str(delay_c.delayCalc(g_path_tm)))
# print("congestion path t:"+str(getCongestion(g_path_tm)))
# print("utilization path t:"+str(getUtilization(g_path_tm)))
# # nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_path_t))
# plotFig(g_path_tm,"PathTm")

# plt.figure(5)
# g = copy.deepcopy(gr)
# opt_delay = g_path_t
# SetBlue(opt_delay,[8])
# print("delay path t:"+str(delay_c.delayCalc(g_path_t)))
# print("congestion path t:"+str(getCongestion(g_path_t)))
# print("utilization path t:"+str(getUtilization(g_path_t)))
# # nx.draw(gr, pos= graphviz_layout(gr, prog='dot'), node_color= colorMap(g_path_t))
# plotFig(opt_delay,"OPT")
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#      gr = p_alg.run_path_alg()
#      print("Dor the KNING")
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
