import math
import matplotlib.pyplot as plt
import SOAR_alg as soar
import networkx as nx
import SMC_alg as smc
import pi_tilde_alg as tilde
import path_alg as p_alg
import delay_clac as delay_c
import PiT_modifide as Pit_m
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import random
# import Delay as dly
import copy

def colorMap(gr):
    color_map = []
    for node in gr.nodes:
        color_map.append(gr.nodes[node]['color'])
    return color_map
def Add_InNetwork_Capacity(g):
    for node in g.nodes:
        att = {node: {'Jobs': {'number': 0, 'list': []}}}
        att[node]['color'] = 'red'
        att[node]['load'] = 0
        nx.set_node_attributes(g, att)


def AvalbiltyCalc(g, cap):
    avalbilty = []
    for node in g.nodes:
        if g.nodes[node]['Jobs']['number'] < cap:
            avalbilty.append(True)
        else:
            avalbilty.append(False)
    return avalbilty

def AvalbiltyCalc_I(Avi, cap):
    avalbilty = []
    for i in Avi:
        if i< cap:
            avalbilty.append(True)
        else:
            avalbilty.append(False)
    return avalbilty


def wieghtFunction(func, i):
    if func == 'linear':
        return 1 + i
    if func == 'power':
        return 1.5 ** i
    if func == 'uniform':
        return 1


def AddWieghtToEges(g, root, func):
    l = list(nx.all_pairs_dijkstra_path_length(g))
    rootIndex = list(g.nodes).index(root)
    depth = max(l[rootIndex][1].values())
    r = {}
    for i in range(0, depth + 1):
        r[i] = []
    for node in g.nodes():
        r[l[rootIndex][1][node]].append(node)

    # for node in r[3]:
    #     nodeRun(g,node,0,2)
    #     print(node)
    # for node in r[2]:
    #     nodeRun(g,node,0,2)
    #     print(node)

    for i in range(0, depth + 1):
        # print('i: '+str(i))
        for node in r[depth - i]:
            # if g.nodes[node]['type'] == 's':
            #     # print(node)
            if node == root:
                return
            perent = list(g.in_edges(nbunch=node))[0][0]
            att = {(perent, node): {'Wieght': wieghtFunction(func, i)}}
            nx.set_edge_attributes(g, att)
def SetLoad(g,load):
    for l in load:
        g.nodes[l[0]]['load']=l[1]

def get_root(g):
    r=[n for n,d in g.in_degree() if d==0]
    return r[0]
def get_blue_ndes(g):
    # gr.nodes[node]['color']='green'
    b = []
    for n in g.nodes():
        if g.nodes[n]['color'] == 'blue':
            b.append(n)
    # print("Blue nodes: " + str(b))
    return b
def getCongestion(g):
    c=[]
    e_list = list(g.edges)
    # e_list.remove(list(g.edges(get_root(g)))[0])
    for e in e_list:
        c.append(sum(g.edges[(e)]['T']))
    return max(c)
def getMinCongestion(g, k, load, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)
    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    availability = AvalbiltyCalc(g, cap)
    return smc.findX(g, root, k, availability)
def getUtilization(g):
    c=[]
    e_list = list(g.edges)
    for e in e_list:
        c.append(sum(g.edges[(e)]['T']))
    return sum(c)
def getPi(g):
    root = get_root(g)
    leafs = leafList(g)
    pi =[]
    for l in leafs:
        pi.append(sum([sum(g.edges[(e)]['T']) for e in list(nx.all_simple_edge_paths(g, source=root, target=l))[0]]))
    return max(pi)

def getPiT(g):
    root = get_root(g)
    leafs =leafList(g)
    pi =[]
    for l in leafs:
        pi.append(sum([sum(g.edges[(e)]['T']) if g.nodes[e[0]]['color'] == 'blue' else 1 for e in list(nx.all_simple_edge_paths(g, source=0, target=l))[0]]))
    return max(pi) + sum(g.edges[(root,0)]['T'])


def getDelay(g):
    return delay_c.delayCalc(g)
def soar_run(g, k, load, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)

    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    availability = AvalbiltyCalc(g, cap)
    soar.gather(g, root, k, availability)
    coloring = soar.color(g, root, root, 0, k)
    return [g, coloring]

def smc_run(g, k, load, A=None, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)

    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    if A == None:
        availability = AvalbiltyCalc(g, cap)
    else:
        availability = AvalbiltyCalc_I(A,cap)
    X = smc.findX(g, root, k, availability)
    smc.run(g, root, k, X, availability)
    coloring = smc.NewColoring(g, root, root, k, X)
    return [g, coloring]

def pi_run(g, k, load):
    SetLoad(g, load)
    X = getMinCongestion(copy.deepcopy(g), k, load)
    # X_star = math.inf
    gt = None
    while gt == None:
        X = 2*X
        gt = p_alg.path_pi_alg(g, int(X), k)
    return gt

def pi_tild_run(gr, k, load, avi= None ,cap = 1):
    if avi == None:
        avi = [0 for i in gr.nodes]
    g=copy.deepcopy(gr)
    SetLoad(g, load)
    X = getMinCongestion(copy.deepcopy(g), k, load)
    print("Pi X: ",X)

    A=AvalbiltyCalc_I(avi,cap)
    # X_star = math.inf
    gt = tilde.pi_tilde_alg(g, int(X), k,A)
    return gt
def piT_mod_run(gr, k, load):
    g=copy.deepcopy(gr)
    SetLoad(g, load)
    X = getMinCongestion(copy.deepcopy(g), k, load)
    print("Pi X: ",X)
    # X_star = math.inf
    gt = Pit_m.piT_mod_alg(g, int(X), k)
    return gt

def paseDistrebution(file):
    distributionFile=open(file)
    distRead=distributionFile.read()
    distRead=distRead.split('\n')
    List=[]
    for line in distRead:
        line=line.replace("{", "")
        line=line.replace("}", "")
        line=line.replace(" ", "")
        line=line.split(":")
        List.append((line[0],line[1].split(",")))
    return List

def leafList(g):
    return [x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)==1]

def removeInEdges(g,node,parent):
    e=list(g.in_edges([node]))
    children=[]
    for i in e:
        if i[0] != parent:
            children.append(i[0])
            g.remove_edge(node,i[0])
    for c in children:
        g= removeInEdges(g,c,node)
    return g
def preferentialAttachment(N,root=0):
    g=nx.barabasi_albert_graph(N,4)

    g = nx.bfs_tree(g, 0)
    g=g.to_directed()
    g=removeInEdges(g,root,root)
    # g=g.reverse()
    return g

def plotFig(g,alg):
    root=len(g.nodes)
    plt.title(alg+" alg ,delay is: "+str(delay_c.delayCalc(g))+" congestion: "+str(getCongestion(g))+
              ", utilization: "+str(getUtilization(g))+r", $\pi_d$: "+str(getPi(g))+r", $\tilde{\pi}_r+\delta_{(r,d)}$: "+str(getPiT(g)))
    pos=graphviz_layout(g,prog='dot')
    node_labels=nx.get_node_attributes(g,'load')
    nx.draw(g,pos,node_color=colorMap(g),font_color="white",labels=node_labels)
    edge_labels = nx.get_edge_attributes(g,'T')
    # dic = {}
    # for x in g.edges():
    #     dic[x] = (sum(g.edges[x]['T']))
    # edge_labels = dic
    nx.draw_networkx_edge_labels(g, pos, edge_labels,label_pos=0.5,rotate=False)

def power_law_distribution(n, alpha, xmin):
    # Generate a uniform distribution
    uniform = np.random.uniform(0, 1, n)

    # Transform the uniform distribution into a power law distribution
    power_law = np.floor((xmin**(1-alpha))*(1-uniform)**(-1/(alpha-1)))

    return power_law.astype(int)

def skewed_distribution(n, skew, mean):
    # Generate a normal distribution
    normal = np.random.normal(mean, 1, n)

    # Transform the normal distribution into a skewed distribution
    skewed = normal + skew * np.abs(normal)

    # Round the values down to the nearest integer
    skewed = np.floor(skewed).astype(int)

    return skewed

def CalcMeanVar(messageExp):
    x=[]
    for i in range(0,len(messageExp[0])):
        y=[]
        for j in range(0,len(messageExp)):
            y.append(messageExp[j][i])
        x.append(y)
    #compute mean and var
    mean=[]
    var=[]
    for i in x:
        mean.append(np.mean(i) )
        var.append(np.sqrt(np.var(i)))
    return [mean,var]

def scale_free_tree(nodes,leafs):
    # Generate a large scale-free graph
    # Generate a random tree
    T = nx.random_tree(255)

    # Check the number of leaf nodes in the tree
    leaf_nodes = [node for node in T if T.degree[node] == 1]

    # Repeat the above steps until the desired number of leaf nodes is reached
    while len(leaf_nodes) != 128:
        T = nx.random_tree(255)
        leaf_nodes = [node for node in T if T.degree[node] == 1]

    return T

def LevelColor(level):
    return [x for x in range(2**level-1,2**(level+1)-1)]
def pickRamdonLevel(level, Avalibily, k):
    x = []
    levelC = LevelColor(level)
    while True:
        for i in levelC:
            if Avalibily[i]:
                x.append(i)
        if len(x) >= k:
            return random.sample(x, k)
        else:
            level = level + 1
            levelC = LevelColor(level)


def LevelJobColor(g,  k, Avalibily):
    level = int(math.log(k, 2))
    levelC = pickRamdonLevel(level, Avalibily, k)
    # levelC=LevelColor(level)
    # flag=True
    # while flag:
    #     count=0
    #     for node in levelC:
    #         if Avalibily[node] == False:
    #             level=level+1
    #             levelC=LevelColor(level)
    #             count=0
    #             break
    #         count=count+1
    #     if count == len(levelC):
    #         flag=False
    g = plot_coloring(g, levelC)
    return [g, levelC]


def TopColor(amount):
    return [x for x in range(0, amount)]


def TopJobColor(g, k, Avalibily):
    top = []
    for node in g.nodes:
        if Avalibily[node]:
            k = k - 1
            top.append(node)
        if k == 0:
            break
    g = plot_coloring(g, top)
    return [g, top]


def MaxColor(amount, g, leafList, Avalibily):
    deg = []
    for leaf in leafList:
        deg.append(g.nodes[leaf]['load'])
    l = []
    for i in leafList:
        if len(l) < amount:
            if Avalibily[leafList[deg.index(max(deg))]]:
                l.append(leafList[deg.index(max(deg))])
            deg[deg.index(max(deg))] = 0
    return l


def MaxJobColor(g, k, Avalibily):
    leafL = leafList(g)
    # MalgC.addLoad(g, load, leafL)
    MaxC = MaxColor(k, g, leafL, Avalibily)
    g = plot_coloring(g, MaxC)
    return [g, MaxC]


def plot_coloring(gr, Blist):
    gt = gr.copy()
    for node in Blist:
        gt.nodes[node]['color'] = 'blue'
    # gr.nodes[0]['color']='green'
    return gt

def level_run(g, k, load,A=None, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)

    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    if A is None:
        availability = AvalbiltyCalc(g, cap)
    else:
        availability = AvalbiltyCalc_I(A, cap)
    gr, c = LevelJobColor(g, k, availability)
    # coloring = smc.NewColoring(g, root, root, k, X)
    return gr
def top_run(g, k, load,A=None, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)

    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    if A is None:
        availability = AvalbiltyCalc(g, cap)
    else:
        availability = AvalbiltyCalc_I(A, cap)
    gr, c = TopJobColor(g, k, availability)
    # coloring = smc.NewColoring(g, root, root, k, X)
    return gr

def max_run(g, k, load,A=None, cap=1):
    # leafL=leafList(g)
    # nodeList, load = readLoad(loadfile)

    root = get_root(g)
    AddWieghtToEges(g, root, 'uniform')
    Add_InNetwork_Capacity(g)
    SetLoad(g, load)
    if A is None:
        availability = AvalbiltyCalc(g, cap)
    else:
        availability = AvalbiltyCalc_I(A, cap)
    gr, c = MaxJobColor(g, k, availability)
    # coloring = smc.NewColoring(g, root, root, k, X)
    return gr