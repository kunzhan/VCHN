import numpy as np
import scipy.sparse as sp
import torch, math, random
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_features(features):
    # """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()



def load_data(dataset, public, percent, seed_k):
	dataset_str = dataset
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))
	x, y, tx, ty, allx, ally, graph = tuple(objects)
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)
	if dataset_str == 'citeseer':
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
		tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range - min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range - min(test_idx_range), :] = ty
		ty = ty_extended
	features = sp.vstack((allx, tx)).tolil()
	labels = np.vstack((ally, ty))
	features[test_idx_reorder, :] = features[test_idx_range, :]
	labels[test_idx_reorder, :] = labels[test_idx_range, :]
	features = preprocess_features(features)
	idx_test_public = test_idx_range.tolist() 
	idx_train, idx_val, idx_test = split_dataset(idx_test_public, len(labels), np.argmax(labels,1), dataset, public, percent, torch.cuda.is_available(), seed_k)
    
	features = torch.FloatTensor(np.array(features.todense()))
	labels = torch.LongTensor(np.argmax(labels,1))
	adj = adj + sp.eye(adj.shape[0])
	adj = normalize_adj(adj)
	adj = sparse_mx_to_torch_sparse_tensor(adj)
	return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def find_pseudo(prediction, t, idx_train, labels):
	prediction = prediction.detach().cpu().numpy()
	idx_train = idx_train.cpu().numpy()
	labels = labels.cpu().numpy()
	
	new_gcn_index = np.argmax(prediction, 1)
	confidence = np.max(prediction, 1)
	sorted_index = np.argsort(-confidence)

	no_class = prediction.shape[1]  # number of class
	t = np.array(np.tile(t, (1,no_class)))
	t = t[0]
	# print('t=',t)
	if hasattr(t, '__getitem__'):
		assert len(t) >= no_class
		index = []
		count = [0 for i in range(no_class)]
		for i in sorted_index:
			for j in range(no_class):
				if new_gcn_index[i] == j and count[j] < t[j] and not (i in idx_train):
					index.append(i)
					count[j] += 1
	else:
		index = sorted_index[:t]

	prediction = new_gcn_index
	prediction[idx_train] = labels[idx_train]

	return torch.LongTensor(index).cuda(), torch.LongTensor(prediction).cuda()	

def common(input1_index, input2_index, input_label1, input_label2):
	input1_index = input1_index.cpu().numpy()
	input2_index = input2_index.cpu().numpy()
	input_label1 = input_label1.cpu().numpy()
	input_label2 = input_label2.cpu().numpy()
	
	common_index = [i for i in input1_index if i in input2_index]
	final_index = [j for j in common_index if input_label1[j]==input_label2[j]]
	return torch.LongTensor(final_index).cuda()
	
def normalize_adj_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def RNM_filter(features, adj, i):
    print("RNM flitering...")
    repeat = i
    adj = adj.to_dense().numpy()
    adj = normalize_adj_adj(adj + np.identity(adj.shape[0]))
    features = features.numpy()
    features = features.astype(np.float32)
    step_transformor = adj
    step_transformor = step_transformor.astype(np.float32)
    for i in range(repeat):
        features = step_transformor.dot(features)
    features = torch.from_numpy(features)
    print("RNM fliter over.")
    return features


def split_dataset(idx_test_public, num_nodes, labels, dataset, public, percent, flag_cuda, seed_k):
    '''test-index，nodes，labels，dataset，1：20 pre class；2：nodes；0：labels rates，GPU'''
    random.seed(seed_k)
    if public == 1:
        if dataset == 'cora':
            idx_train, idx_val, idx_test = range(140), range(140, 640), idx_test_public
        elif dataset == 'citeseer':
            idx_train, idx_val, idx_test = range(120), range(120, 620), idx_test_public
        elif dataset == 'pubmed':
            idx_train, idx_val, idx_test = range(60), range(60, 560), idx_test_public
    elif public == 0:
        all_data, all_class = np.arange(num_nodes).astype(int), np.unique(labels)
        idx_train, idx_val, idx_test = [], [], []
        for c in all_class:
            idx_train = np.hstack([idx_train,random.sample(list(np.where(labels == c)[0].astype(int)), math.ceil(np.where(labels==c)[0].shape[0]*percent))])
        others = np.delete(all_data.astype(int), idx_train.astype(int))
        for c in all_class:
            idx_val = np.hstack([idx_val,random.sample(list(np.where(labels[others] == c)[0].astype(int)), math.ceil(500/all_class.shape[0]) )])
        others = np.delete(others.astype(int), idx_val.astype(int))
        for c in all_class:
            idx_test = np.hstack([idx_test,random.sample(list(np.where(labels[others] == c)[0].astype(int)), min(math.ceil(1000/all_class.shape[0]), np.where(labels[others]==c)[0].astype(int).shape[0]))])
    if flag_cuda:
        idx_train, idx_val, idx_test = torch.LongTensor(idx_train).cuda(), torch.LongTensor(idx_val).cuda(), torch.LongTensor(idx_test).cuda()
    return idx_train, idx_val, idx_test