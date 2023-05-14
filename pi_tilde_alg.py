
import networkx as nx
import matplotlib.pyplot as plt
import threading
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.ticker as mticker
import math
# import os
import numpy as np
# import copy
import time
from tqdm import tqdm
import pandas as pd


def SetLoad(g, load):
    for l in load:
        g.nodes[l[0]]['load'] = l[1]






# def mChild(g, x, i, n, col):
#     data = g.nodes[n]['beta']
#     x = int(x)
#     if col == 'blue':
#         tmp = [data.at[data.loc[(data['x'] == x_t) & (data['k'] == i)].index[0], 'beta'] + x_t for x_t in range(0, x + 1)]
#         for t in tmp:
#             if t <= x:
#                 return data.at[data.loc[(data['x'] == tmp.index(t)) & (data['k'] == i)].index[0], 'beta']
#     else:
#         tmp = [data.loc[(data['x'] == x_t)& (data['k'] == i)]['beta'].item() if 1+x_t <= x else math.inf for x_t in range(0, x + 1)]
#         # tmp = [x_t + 1 if data.at[data.loc[(data['x'] == x_t) & (data['k'] == i)].index[0], 'beta']< math.inf else math.inf for x_t in range(0, x + 1)]
#         return min(tmp)
#     return np.inf
def mChild(g, x, i, n, col):
    data = g.nodes[n]['beta']
    x = int(x)
    if col == 'blue':
        tmp=[data[x_t][i]+x_t for x_t in range(0, x)]
        if tmp and min(tmp) <= x:
            return 1
    else:
        tmp =[data[x_t][i] for x_t in range(0, x)]
        if tmp:
            return min(tmp)
    return np.inf

def nodeGather(g, node, X, k,A):
    try:
        load = g.nodes[node]['load']  # Note to take care
    except:
        load = 0
    C = [x for x in list(g.neighbors(node))]
    if 'beta' in g.nodes[node]:
        s=len(g.nodes[node]['beta'])
    else:
        att={node: {'beta':{},'beta_m':{},'node':node,'children':[],'color':'red'}}
        nx.set_node_attributes(g, att)
        s=0
    beta=g.nodes[node]['beta']
    if not C:#leaf node
        for x in range(s,X+1):
            beta[x]=[load]
            for i in range(1,k+1):
                if A[node]:
                    beta[x].append(1)
                else:
                    beta[x].append(np.inf)
        g.nodes[node]['beta'] = beta
        return g
    beta_m=g.nodes[node]['beta_m']
    first = True
    for child in C:
        if not child in beta_m.keys():
            beta_m[child]={}
        # print("node: ",node,"child: ",child)
        if child not in g.nodes[node]['children']:
            g.nodes[node]['children'].append(child)
        for x in range(s, X + 1):
            beta_m[child][x]={'red':[],'blue':[]}
            for i in range(0, k + 1):
                # print('x: '+str(x)+' i: '+str(i))
                if child == C[0]:  # first child m0

                    if i > 0 and A[node] and mChild(g, x, i - 1, child, 'blue') < np.inf :
                        beta_m[child][x]['blue'].append(1)
                        # 'x','k','m','blue','red'
                        # data.loc[len(data.index)] = [x, i, C.index(child), 1, np.nan]

                    else:
                        beta_m[child][x]['blue'].append(np.inf)
                        # 'x','k','m','blue','red',
                        # data.loc[len(data.index)] = [x, i, C.index(child), np.inf, np.nan]
                        # 'x','k','m','blue','red'

                    beta_m[child][x]['red'].append(mChild(g, x, i, child, 'red') + load)
                else:  # not first child
                    if i > 0 and A[node] and np.min([beta_m[C[C.index(child)-1]][x]['blue'][i-j]+mChild(g, x, j, child, 'blue') for j in range(0,i)]) < np.inf:
                        beta_m[child][x]['blue'].append(1)
                    else:
                        # 'x','k','m','blue','red'
                        beta_m[child][x]['blue'].append(np.inf)
                        # data.at[data.loc[(data['x']==x)&(data['k']==i)&(data['m']==C.index(child))].index[0],'blue']=math.inf
                    beta_m[child][x]['red'].append(np.min([beta_m[C[C.index(child)-1]][x]['red'][i-j]+mChild(g, x, j, child, 'red') for j in range(0,i+1)]))
    g.nodes[node]['beta_m'] = beta_m
    for x in range(s, X + 1):
        beta[x] =[beta_m[child][x]['red'][0]]
        for i in range(1, k + 1):
            beta[x].append(np.min([beta_m[child][x]['red'][i],beta_m[child][x]['blue'][i]]))
                # data.at[data.loc[(data['x']==x)&(data['k']==i)&(data['m']==C.index(child))].index[0],'beta']=tmp
    g.nodes[node]['beta'] = beta

    return g


def Gather(g, X, k,A):
    # if not ('beta' in g.nodes[0].keys()):
    #     for n in g.nodes:
    #         att = {n: {'color': 'red', 'data': pd.DataFrame(columns=['x', 'k', 'm', 'blue', 'red']),
    #                    'beta': pd.DataFrame(columns=['x', 'k', 'beta','col']), 'children': []}}
    #         nx.set_node_attributes(g, att)
    order = list(nx.bfs_tree(g, 0))
    order.reverse()
    start_time = time.time()
    for node in tqdm(order):
        # print('node '+str(node))
        nodeGather(g, node, X, k,A)
    print("Gather Running time: " + str(time.time() - start_time))
    return g


def nodeThread(*args):
    # print("node " +str(args[1])+' thread')
    # print(args)
    nodeGather(args[0], args[1], args[2], args[3])
    # print("node " +str(node)+' thread ended')


def GatherThread(g, X, k):
    start_time = time.time()

    if not ('beta' in g.nodes[0].keys()):
        for n in g.nodes:
            att = {n: {'color': 'red', 'data': pd.DataFrame(columns=['x', 'k', 'm', 'blue', 'red']),
                       'beta': pd.DataFrame(columns=['x', 'k', 'beta']), 'children': []}}
            nx.set_node_attributes(g, att)
    l = list(nx.all_pairs_dijkstra_path_length(g))
    rootIndex = list(g.nodes).index(len(g.nodes) - 1)
    depth = max(l[rootIndex][1].values())

    r = {}
    for i in range(0, depth + 1):
        r[i] = []
    for node in g.nodes():
        r[l[rootIndex][1][node]].append(node)

    for i in tqdm(range(0, depth + 1)):
        threads = []
        # level_time = time.time()
        # print('deg:' + str(2) + 'i: ' + str(i))
        for node in r[depth - i]:
            # print(node)

            # print("node call"+str(node))
            t = threading.Thread(target=nodeThread, args=(g, node, X, k,))
            threads.append(t)
        for x in threads:
            x.start()
        for x in threads:
            x.join()
        # print ("round " +str(i)+' finished')
        # print("level time: "+str(time.time()-level_time))

    print("Gather Running time: " + str(time.time() - start_time))
    return g


def color(g, node, x, i):
    g.nodes[node]['color'] = 'red'
    beta_m = g.nodes[node]['beta_m']
    beta = g.nodes[node]['beta']
    C = g.nodes[node]['children']
    if C == [] :  # leaf node
        if i>0:
            g.nodes[node]['color'] = 'blue'
        return
    if i > 0:
        # print(str(beta_m[list(beta_m)[-1]][x]['blue'][i])+'  '+str(beta_m[list(beta_m)[-1]][x]['red'][i]))
        if beta_m[list(beta_m)[-1]][x]['blue'][i] < beta_m[list(beta_m)[-1]][x]['red'][i]:
            g.nodes[node]['color'] = 'blue'
        for child in C[::-1]:
            # print("node: ",node,"child: ",c)
            if child == C[0]:  # first child
                if g.nodes[node]['color'] == 'blue':
                    # print("blue i: ",i)
                    x_1 = X_m(g, child, x, i-1)
                    color(g, child, x_1, i - 1)
                    return
                else:
                    color(g, child, x-1, i)
                    return
            # print("node: "+str(node)+" child:" +str(child))
            # print('j arg: ' +str([beta_m[C[C.index(child) - 1]][x][g.nodes[node]['color']][i - j] + mChild(g, x, j, child, g.nodes[node]['color']) for j in range(0, i + 1)]))
            j = np.argmin([beta_m[C[C.index(child) - 1]][x][g.nodes[node]['color']][i - j] + mChild(g, x, j, child, g.nodes[node]['color']) for j in range(0, i + 1)])
            if g.nodes[node]['color'] == 'blue':
                x_m = X_m(g,child,x,j)
            else:
                x_m=x-1

            # print("node: ",node," c: ",c,"i: ",i," x': ",x," j: ",j)
            color(g, child, x_m, j)
            i = i - j
    return


def leafList(g):
    return [x for x in g.nodes() if g.out_degree(x) == 0 and g.in_degree(x) == 1]


# def X_Star(g, n, i):
#     beta = g.nodes[n]['beta']
#     lst = list(beta.loc[(beta['k'] == i)].index)
#     return min([beta.loc[(beta['k'] == i)]['beta'][l] + 1 if beta.loc[beta['k'] == i]['col'][l] ==0 else beta.loc[(beta['k'] == i)]['beta'][l] + beta.loc[beta['k'] == i]['x'][l] for l in lst])

def X_Star(g, n, i):
    beta = g.nodes[n]['beta']
    return np.argmin([beta[x][i] + x for x in beta.keys()])
def X_m(g, n, x,i):
    beta = g.nodes[n]['beta']

    return np.argmin([beta[x_t][i] + x_t for x_t in range(0, x+1)])

def X_Tag(g, n, x, i,col):
    beta = g.nodes[n]['beta']
    lst = list(beta.loc[(beta['k'] == i)].index)
    if col == "blue":
    # return min([beta.loc[(beta['k'] == i)]['beta'][l] + beta.loc[beta['k'] == i]['x'][l] for l in lst])
        pi=[beta.loc[(beta['k'] == i)]['beta'][l] + beta.loc[beta['k'] == i]['x'][l] for l in lst]
        b =[]
        for j in range(0,len(pi)):
            if pi[j] == min(pi):
                b.append(beta.loc[(beta['k'] == i)]['beta'][lst[j]])
        return min(pi) - min(b) #beta.loc[beta['k'] == i]['x'][lst[pi.index(min(pi))]]
    else:
        return x-1

def minX(g, n, i, X, color):
    beta = g.nodes[n]['beta']
    lst = list(beta.loc[(beta['k'] == i)].index)
    if color == 'blue':
        tmp = [beta.loc[(beta['k'] == i)]['beta'][l] + beta.loc[beta['k'] == i]['x'][l] for l in lst]
    else:
        tmp = [beta.loc[(beta['k'] == i)]['beta'][l] + 1 for l in lst]
    for t in tmp:
        if t <= X:
            inDex = tmp.index(t)
            return [beta.loc[(beta['k'] == i)]['beta'][lst[inDex]], int(beta.loc[(beta['k'] == i)]['x'][lst[inDex]])]


def pi_tilde_alg(g, X, i,A):
    x_star =  np.inf
    # X= sum([g.nodes[l]['load'] for l in leafList(g)])
    h=nx.algorithms.dag.dag_longest_path_length(g, weight=None)
    X = X+  h # int(math.ceil(1.2 * X)) #cong +h
    while (x_star == np.inf) or (int(len(g.nodes[0]['beta']) ) < (x_star+g.nodes[0]['beta'][x_star][i])): # len(x) >= x*+beta

        g = Gather(g, X, i,A)
        # g = GatherThread(g, X, i)
        x_star = X_Star(g, 0, i)
        # print(r"$x^*$: " + str(x_star))
        # print("x_star+g.nodes[0]['beta'][x_star][i]: "+str(x_star + g.nodes[0]['beta'][x_star][i])+"len(g.nodes[0]['beta']"+str(len(g.nodes[0]['beta']))+" "+str((len(g.nodes[0]['beta'])  < x_star+g.nodes[0]['beta'][x_star][i])))
        X = X + h
    if x_star < np.inf :
        for n in g.nodes():
            g.nodes[n]['color'] = 'red'
        print(r"$x^*$: "+str(x_star))
        color(g, 0, x_star, i)
        return g
    else:
        return

# def run_path_alg():
#     deg = 2
#     h = 3
#     load = [(7, 5), (8, 4), (9, 5), (10, 1), (11, 4), (12, 11), (13, 6), (14, 6)]
#     k = 8
#     X = 15
#     g = nx.balanced_tree(deg, h, create_using=nx.DiGraph)
#     # root=len(g.nodes)
#     # g.add_edge(len(g.nodes),0)
#
#     # # for n in g.nodes:
#     # #     att={n:{'color':'red','load':0,'data':pd.DataFrame(columns=['x','k','m','blue','red']),'beta':pd.DataFrame(columns=['x','k','beta']),'children':[]}}
#     # #     nx.set_node_attributes(g,att)
#
#     # # leafL=leafList(g)
#     # # # load=[(leafL[i],1) for i in range(0,len(leafL))]
#
#     SetLoad(g, load)
#     g = Gather(g, X, k)
#     # X = 2
#     g = Gather(g, X, k)
#     # threadrun(gr,root,k,X,Avilabilty)
#     x_star = X_Star(g, 0, k)
#     if x_star < np.inf:
#         color(g, 0, k, x_star)
#     else:
#         print("no sol!!!")
#     X = 30
#     g = Gather(g, X, k)
#     # threadrun(gr,root,k,X,Avilabilty)
#     x_star = X_Star(g, 0, k)
#     if x_star < np.inf:
#         color(g, 0, k, x_star)
#     else:
#         print("no sol!!!")
#     k = 8
#     # g=nx.read_gpickle("new_test.gpickle")
#     x_star = X_Star(g, 0, k)
#     color(g, 0, k, x_star)
#     pos = graphviz_layout(g, prog='dot')
#     # node_labels=nx.get_node_attributes(g,'load')
#     nx.draw(g, pos, node_color= colorMap(g), with_labels=True)
#     plt.show()
#     return g

