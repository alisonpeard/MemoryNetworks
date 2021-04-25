

import numpy as np
import pathpy as pp
import scipy as sp
from collections import defaultdict # think these two are used to iterate through dicts
from collections.abc import Iterable

def matprint(mat, fmt="g"):
    # just some useful code to displays matrices nicer for looking at this
    # https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
  
# Try it! 

#---------------------------------------------------------------------------------------------------------
# directed network processing

def make_connected(net):
    '''Adds one out-link to a randomly selected node to all hanging nodes'''
    nodes_list = list(net.nodes.keys())
    out_degrees = np.array(net.node_properties('outdegree'))
    ind = np.where(out_degrees == 0)[0]
    hanging_nodes = []
    for i in ind: hanging_nodes.append(list(net.nodes.keys())[i])
    for node in hanging_nodes : net.add_edge(node,np.random.choice(nodes_list))
    return net

def transition_mat(net,method="smart",alpha=0.85):
    '''modified from pathpy pagerank code, transition matrix with teleportation
    this will slow down computation as matrices will no longer be sparse'''
    
    N = net.ncount()
    I = sp.sparse.identity(N)
    A = net.adjacency_matrix()

    # row sums are out-degrees for adjacency matrix
    row_sums = np.array(A.sum(axis=1)).flatten()

    # replace non-zero entries x by 1/x
    row_sums[row_sums != 0] = 1.0 / row_sums[row_sums != 0]

    # indices of zero entries in row_sums
    b = list(np.where(row_sums != 0)[0])
    d = list(np.where(row_sums == 0)[0])

    # create sparse matrix with row_sums as diagonal elements
    Dinv = sp.sparse.spdiags(row_sums.T, 0, A.shape[0], A.shape[1],
                               format='csr')

    # with this, we have divided elements in non-zero rows in A by 1 over the row sum
    T = Dinv * A
    
    if method == "smart":
        # calculate preference vector using node in-strengths
        w_in = np.array(net.node_properties("inweight"))
        W = sum(w_in)
        v = w_in/W 
    elif method == "standard":
        v = np.ones(N)/N
    
    # replace nonzero rows with alpha*T + (1-alpha)*u
    for ib in b: T[ib,:] = alpha*T[ib,:] + (1-alpha)*v
        
    # replace all fully zero rows with v 
    for id in d: T[id,:] = v
    
    return T.todense()

def get_stationary(T):
    '''will get the left eigenvector corresponding to eigenvalue 1 and return it normalised'''
    # np.linalg.eig finds right eigenvectors - code from a very helpful stack exchange
    # https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain?fbclid=IwAR0YnlQ7iwr1Ve1kRn-b6CDT0rbq7lDdBM1oD_KwiaEODWPv_-GMwiGBcVw
    evals, evecs = np.linalg.eig(T.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    stationary = np.squeeze(np.asarray(stationary))
    return stationary


def Laplacian_mat(net,alpha=0.85):
    '''random walk-normalised Laplacian with smart recorded teleportation: fix this to match papers and check applicability'''
    N = net.ncount()
    I = np.eye(N)
    T = transition_mat(net)
    
    Lrw = I - T 
    
    return Lrw

def my_pagerank(net,alpha=0.85,method = "smart"):
    '''Needs to be imporoved to be more efficient but does the job right now
    slightly modified.'''
    
    pr = defaultdict(lambda: 0)
    
    # adjacency matrix where Aij s.t. i->j
    I = sp.sparse.identity(net.ncount())
    A = net.adjacency_matrix()
    T = transition_mat(net,method=method)
    
    pr = get_stationary(T)
    
    pr = dict(zip(net.nodes, map(float, pr)))
    
    return pr
#---------------------------------------------------------------------------------------------------------
# core-periphery stuff

def get_node_strengths(net):
    strengths = np.array(net.node_properties('inweight')) + np.array(net.node_properties('outweight'))
    return strengths

def remove_hanging(net,method="cut"):
    '''Remove hanging nodes and returns network'''
    if method == "cut":
        out_degrees = np.array(net.node_properties('outweight'))
        
        if (out_degrees == 0).any():
            ind = np.where(out_degrees == 0)[0] 
            hanging_nodes = []
            for i in ind: hanging_nodes.append(list(net.nodes.keys())[i])
            for node in hanging_nodes : net.remove_node(node)
            
        return net
    
    else:
        return "no method defined for this"

    
def alpha_S(ind, T_t, ps_t):
    '''Calculate the persistence probability of a set of nodes S 
    ind : list of indices
    T_t : sparse transposed transition matrix
    ps_t = stationary probability column vector'''
    T_t = T_t[ind,:][:,ind]
    ps_t = ps_t[ind]
    return T_t.dot(ps_t).sum()/ps_t.sum()
    
    
# tidy this later
# get rid of for loop because of other thing
def get_coreness(net,R=1):
    '''Calculates coreness of each node R times and returns network with new 'coreness' node attributes.
    Network must be ergodic and have no hanging nodes.'''
    
    A = net.adjacency_matrix(weighted=True)
    T_t = transition_mat(net)
    #T_t = net.transition_matrix()
    #T = T_t.transpose()
    N = net.ncount()
    
    #out_degrees = np.array(net.node_properties('outweight'))
    #assert ((out_degrees > 0).all()), "Network must be ergodic."
    node_strengths = get_node_strengths(net)
    
    # get stationary distribution using augmented transition matrix
    ps_t = np.array(list(my_pagerank(net).values()))
    
    # previously
    #v11 = -1
    #while v11<0:
    #    _,ps = .sparse.linalg.eigs(T,k=1,which='LM',tol=0) # check L&R
    #    ps_t = ps.real
    #    v11 = ps_t[0]
          
    
    # get indices for node names (quick fix for now)
    nodes_ind = np.arange(0,N)
    node_dict = {}
    coreness = {}
    i=0
    for node in net.nodes.keys():
        node_dict.update({i:node})
        coreness.update({node:0.0}) # initialise to zero
        i=i+1
    
    
    # perform R runs of the CP algorithm - don't need this anymore so fix and remove R=1 stuff
    r=0
    while r < R:
        #print(r/R)
        # calculate node strengths
        i_min = np.where(node_strengths == node_strengths.min())[0] 
        i_min = np.random.choice(i_min,1)[0] # randomly choose one if there's multiple mins
        coreness[node_dict[i_min]] = coreness[node_dict[i_min]] + 0.0
        s0 = list(net.nodes.keys())[i_min] # node to initalise CP algorithm
        
        # (greedy) algorithm to test which node will create smallest increase in alpha_S
        S = [i_min]
        alpha = alpha_S(S,T_t,ps_t) # alpha_S is zero for any single node
        nodes = np.delete(nodes_ind,i_min)

        while (len(nodes)>0):
            alpha_test = np.empty(N)
            alpha_test[:] = np.nan
            
            # calculate alpha increase for each node
            for node in nodes:
                S_test = S + [node]
                alpha_test[node] = alpha_S(S_test,T_t,ps_t) 

            # choose minimum alpha for this step
            node_min = np.where(alpha_test == np.nanmin(alpha_test))[0] 
            node_min = np.random.choice(node_min,1)[0]
            alpha = alpha + alpha_test[node_min]
            coreness[node_dict[node_min]] = coreness[node_dict[node_min]] + alpha_test[node_min]
    
            S = S + [node_min]
            nodes = nodes[nodes!=node_min]
        
        r = r+1
        
    coreness.update({v:u/R for v,u in coreness.items()})
    
    for node, val in coreness.items():
        net.nodes[node]['coreness']=val
    
    return net,coreness

def print_coreness(net,coreness,alpha_c=1e-3,col1='darkorange',col2='lightskyblue'):
    '''https://github.com/IngoScholtes/csh2018-tutorial/blob/master/solutions/2_pathpy.ipynb'''
    edge_coreness = {} # colour edges according to coreness of nodes they emanate from
    for edge in net.edges.keys():
        edge_coreness.update({edge:coreness[edge[0]]})
    
    style={}
    style['edge_color']={v:col1 if u < alpha_c else col2 for v,u in edge_coreness.items()}
    style['node_color']={v:col1 if u < alpha_c else col2 for v,u in coreness.items()}
    style['force_charge']={v: -20 if u<alpha_c else -20 for v,u in coreness.items()}
    pp.visualisation.plot(net, **style)

#---------------------------------------------------------------------------------------------------------
# m2 core-periphery stuff