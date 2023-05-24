import networkx as nx
from networkit import *
import random
import pickle
import numpy as np
import time
from datetime import datetime
import glob
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import scipy.sparse as sp
import copy
import torch



def log(val,logfile):
    with open(logfile,'a') as f:
        f.write(f"\n{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}: {val}")

def create_graph(graph_type,min_nodes,max_nodes):

    num_nodes = np.random.randint(min_nodes,max_nodes)

    if graph_type == "ER":
        #Erdos-Renyi random graphs
        p = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes,p = p,directed = True)
        return g_nx

    if graph_type == "SF":
        #Scalefree graphs
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        return g_nx


    if graph_type == "GRP":
        #Gaussian-Random Partition Graphs
        s = np.random.randint(200,1000)
        v = np.random.randint(200,1000)
        p_in = np.random.randint(2,25)*0.0001
        p_out = np.random.randint(2,25)*0.0001
        g_nx = nx.generators.gaussian_random_partition_graph(num_nodes,s = s, v = v, p_in = p_in, p_out = p_out, directed = True)
        assert nx.is_directed(g_nx)==True,"Not directed"
        return g_nx


def get_out_edges(g_nkit,node_sequence):
    global all_out_dict
    all_out_dict = dict()
    for all_n in node_sequence:
        all_out_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forEdgesOf(all_n,nkit_outedges)
            
    return all_out_dict

def get_in_edges(g_nkit,node_sequence):
    global all_in_dict
    all_in_dict = dict()
    for all_n in node_sequence:
        all_in_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forInEdgesOf(all_n,nkit_inedges)
            
    return all_in_dict


def nkit_inedges(u,v,weight,edgeid):
    all_in_dict[u].add(v)


def nkit_outedges(u,v,weight,edgeid):
    all_out_dict[u].add(v)


def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    assert g_nx.number_of_nodes()==g_nkit.numberOfNodes(),"Number of nodes not matching"
    assert g_nx.number_of_edges()==g_nkit.numberOfEdges(),"Number of edges not matching"
        
    return g_nkit


def clique_check(index,node_sequence,all_out_dict,all_in_dict):
    node = node_sequence[index]
    in_nodes = all_in_dict[node]
    out_nodes = all_out_dict[node]

    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if tmp_out_nodes.issubset(all_out_dict[in_n]) == False:
            return False
    

    return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def cal_exact_bet(g_nkit):

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict


def cal_exact_degree(g_nkit):

    exact_deg = centrality.DegreeCentrality(g_nkit,normalized=False).run().ranking()
    exact_deg_dict = dict()
    for j in exact_deg:
        exact_deg_dict[j[0]] = j[1]
    return exact_deg_dict


def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data,num_copies,adj_size):

    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num),dtype=np.float)
    deg_mat = np.zeros((adj_size,total_num),dtype=np.float)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0

    for g_data in list_data:

        graph, cent_dict, deg_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
                deg_mat[ind,mat_index] = deg_dict[node]
            mat_index +=  1

    serial_list = [i for i in range(total_num)]
    random.shuffle(serial_list)

    list_graph = reorder_list(list_graph,serial_list)
    list_n_sequence = reorder_list(list_n_sequence,serial_list)
    list_node_num = reorder_list(list_node_num,serial_list)
    cent_mat_tmp = cent_mat[:,np.array(serial_list)]
    cent_mat = cent_mat_tmp
    deg_mat_tmp = deg_mat[:,np.array(serial_list)]
    deg_mat = deg_mat_tmp
    return list_graph, list_n_sequence, list_node_num, cent_mat, deg_mat


def graph_to_adj_bet(list_graph,list_n_sequence,list_node_num,model_size,showing_zeros = False):
    
    list_adjacency = list()
    list_adjacency_t = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    print(f"Processing {len(list_graph)} graphs...")
    for i in range(len(list_graph)):
        #print(f"Processing graphs: {i+1}/{len(list_graph)}")
        
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        #self_loops = [i for i in graph.selfloop_edges()]
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)
        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        total_zero_beforehand = node_num-len(non_zero_ind)
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)


        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
                total_zero_beforehand += 1
                degree_arr[index,0]=0.0

        if showing_zeros:
            print(f"Graph {i}, Nodes: {node_num}, Beforehand zeros: {total_zero_beforehand} ({(total_zero_beforehand/node_num)*100} %)")

        adj_temp = adj_temp.multiply(csr_matrix(degree_arr))
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))
                

        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor

        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_t = csr_matrix(adj_temp_t)
        adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))
        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)
        list_adjacency_t.append(adj_mat_t)
       
    return list_adjacency,list_adjacency_t




def ranking_correlation(y_out,true_val,node_num,model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    return kt


def loss_cal(y_out,true_val,num_nodes,device,model_size):

    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    
    _,order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes*20

    ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    

    rank_measure=torch.sign(-1*(ind_1-ind_2)).float()
        
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
        

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
 
    return loss_rank





def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train,model,device,optimizer,size):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()


def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,deg_mat_test,model,device,size):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        deg_arr = torch.from_numpy(deg_mat_test[:,j]).float()
        deg_val = deg_arr.to(device)  

        kt = ranking_correlation(y_out,true_val,num_nodes,size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")
    return {"kt":np.mean(np.array(list_kt)), "std":np.std(np.array(list_kt))}


def nozeros(a,b):
    aout = []
    bout = []
    
    for idx,v in enumerate(b):
        if v != 0:
            aout.append(a[idx])
            bout.append(b[idx])
    
    return aout,bout


def test_nozeros(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,deg_mat_test,model,device,size):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        deg_arr = torch.from_numpy(deg_mat_test[:,j]).float()
        deg_val = deg_arr.to(device)  

        y_out = y_out.reshape((size))
        true_val = true_val.reshape((size))
        predict_arr = y_out.cpu().detach().numpy()
        true_arr = true_val.cpu().detach().numpy()
        
        a,b = nozeros(predict_arr[:num_nodes],true_arr[:num_nodes])
        kt,_ = kendalltau(a,b)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")

    print(f"   Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}")
    return {"kt":np.mean(np.array(list_kt)), "std":np.std(np.array(list_kt))}



def test_onegraph(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,deg_mat_test,model,device,size):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)

    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        deg_arr = torch.from_numpy(deg_mat_test[:,j]).float()
        deg_val = deg_arr.to(device) 

        y_out = y_out.reshape((size))
        true_val = true_val.reshape((size))
        deg_val = deg_val.reshape((size))

        pred = y_out.cpu().detach().numpy()
        real = true_val.cpu().detach().numpy()
        deg = deg_val.cpu().detach().numpy()

        return {'pred': pred[:num_nodes], 'true': real[:num_nodes],'deg': deg[:num_nodes],'kt': kendalltau(pred[:num_nodes],real[:num_nodes])[0]}


def test_onegraph_nozeros(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test,deg_mat_test,model,device,size):
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)

    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)
    
        
        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        deg_arr = torch.from_numpy(deg_mat_test[:,j]).float()
        deg_val = deg_arr.to(device) 

        y_out = y_out.reshape((size))
        true_val = true_val.reshape((size))
        deg_val = deg_val.reshape((size))

        pred = y_out.cpu().detach().numpy()
        real = true_val.cpu().detach().numpy()
        deg = deg_val.cpu().detach().numpy()

        a,b = nozeros(pred[:num_nodes],real[:num_nodes])
        kt,_ = kendalltau(a,b)
        list_kt.append(kt)

        return {'pred': pred[:num_nodes], 'true': real[:num_nodes],'deg': deg[:num_nodes],'kt': kt}