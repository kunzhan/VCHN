from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import accuracy, load_data, RNM_filter, find_pseudo, common
from models import GVCLN
import warnings
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type = bool, default=True, help='Use CUDA for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_1', type=int, default=16, help='Number of hidden units of Network_1.')
parser.add_argument('--dropout_1', type=float, default=0.5, help='Dropout rate of Network_1.')
parser.add_argument('--hidden_2', type=int, default=16, help='Number of hidden units of Network_2.')
parser.add_argument('--nb_heads_2',type=int, default=3, help='Number of head of Network_2.')
parser.add_argument('--dropout_2', type=float, default=0.6, help='Dropout rate of Network_2.')
parser.add_argument('--alpha_2', type=float, default=0.2, help='Alpha for the leaky_relu of Network_2.')
parser.add_argument('--dataset', type=str, default='cora', help = 'Dataset: cora or citeseer or pubmed')
parser.add_argument('--public', type = int, default = 0, help = '1 for 20 pre class and 0 for label rate')
parser.add_argument('--fastmode', type=bool, default=False, help='Use validation set or not.')
parser.add_argument('--percent', type=float, default=0.005, help='Label rate.')
parser.add_argument('--patience', type=int, default=100, help='Patience for early stop.')
parser.add_argument('--t1', type=int, default=200, help='First step pseudo label.')
parser.add_argument('--t2', type=int, default=300, help='Second step pseudo label.')
parser.add_argument('--k', type=int, default=15, help='RNM filter parameter.') 

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print("Parameter settings:",args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

def Graph_pseudo(adj, features, labels, idx_train, idx_val, idx_test):
	if args.cuda:
		features = features.cuda()
		adj = adj.cuda()
		labels = labels.cuda()
	model = GVCLN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid_1=args.hidden_1, \
				  dropout_1=args.dropout_1, nhid_2=args.hidden_2, dropout_2=args.dropout_2, alpha_2=args.alpha_2, \
				  nheads_2=args.nb_heads_2)
	optim_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model.cuda()
	
	bad_counter = 0 #Count the bad epochs
	vacc_mx_1 = 0.0 #Network_1's max verification or training accuracy 
	vacc_mx_2 = 0.0 #Network_2's max verification or training accuracy 
	loss_min_1 = 1000000000.0 #Network_1's min loss
	loss_min_2 = 1000000000.0 #Network_2's min loss
	best_acc = 0.0 #Record the test accuracy of the optimal model
	t = args.t1 #Number of pseudo labels
	for epoch in range(args.epochs):
		#Find common pseudo labels
		model.eval()
		model_one_output, model_two_output, loss_11, loss_21, loss_12, loss_22 = model(features, adj, idx_train, labels)
		model_one_index, model_one_prediction = find_pseudo(F.softmax(model_one_output, 1), t, idx_train, labels)
		model_two_index, model_two_prediction = find_pseudo(F.softmax(model_two_output, 1), t, idx_train, labels)
		model_common_index = common(model_one_index, model_two_index, model_one_prediction, model_two_prediction)
		num = len(model_common_index)

		#Training
		model.train()
		optim_model.zero_grad()
		model_one_output, model_two_output, loss_11, loss_21, loss_12, loss_22 = model(features, adj, idx_train, labels)
		pseudo_loss_1 = torch.nn.CrossEntropyLoss()(model_one_output[model_common_index], model_two_prediction[model_common_index])
		pseudo_loss_2 = torch.nn.CrossEntropyLoss()(model_two_output[model_common_index], model_one_prediction[model_common_index])
		if epoch < 100 :
			loss_1 = loss_11
			loss_2 = loss_21
		else:
			loss_1 = loss_12 + ((epoch-100)/500)*pseudo_loss_1
			loss_2 = loss_22 + ((epoch-100)/500)*pseudo_loss_2
		loss_1.backward(retain_graph=True)
		loss_2.backward()
		optim_model.step()

		model.eval()
		model_one_output, model_two_output, loss_11, loss_21, loss_12, loss_22 = model(features, adj, idx_train, labels)
		#Validation
		if not args.fastmode:
			model_one_val = accuracy(F.softmax(model_one_output[idx_val], 1), labels[idx_val])
			model_two_val = accuracy(F.softmax(model_two_output[idx_val], 1), labels[idx_val])
			model_one_test = accuracy(F.softmax(model_one_output[idx_test], 1), labels[idx_test])
			model_two_test = accuracy(F.softmax(model_two_output[idx_test], 1), labels[idx_test])

			if model_one_val.item() >= vacc_mx_1 and loss_11.item() <= loss_min_1 :
				vacc_mx_1 = np.max((model_one_val.item(), vacc_mx_1))
				loss_min_1 = np.min((loss_11.item(), loss_min_1))
				bad_counter = 0
				best_acc_1 = model_one_test.item()
			else:
				bad_counter += 1

			if model_two_val.item() >= vacc_mx_2 and loss_21.item() <= loss_min_2 :
				vacc_mx_2 = np.max((model_two_val.item(), vacc_mx_2))
				loss_min_2 = np.min((loss_21.item(), loss_min_2))
				bad_counter = 0
				best_acc_2 = model_two_test.item()
			else:
				bad_counter += 1

			if vacc_mx_1 > vacc_mx_2 :
				best_acc = best_acc_1
			else :
				best_acc = best_acc_2

			print("\r\rEpoch:%04d || " % epoch, "Val_acc_one:%5.4f%% || " % (model_one_val.item()*100), \
				 "Val_acc_two:%5.4f%% || " % (model_two_val.item()*100), \
				 "Best_acc:%5.4f%% || " % (best_acc*100), "Pseudo_labels:%04d || " % num, \
				 "Bad_counter:%02d" % bad_counter, end='')

			if bad_counter >= args.patience and t == args.t1 :
				bad_counter = 0
				t = args.t2
			if bad_counter >= args.patience :
				break

		else:
			train_acc1 = accuracy(F.softmax(model_one_output[idx_train], 1), labels[idx_train])
			train_acc2 = accuracy(F.softmax(model_two_output[idx_train], 1), labels[idx_train])
			model_one_test = accuracy(F.softmax(model_one_output[idx_test], 1), labels[idx_test])
			model_two_test = accuracy(F.softmax(model_two_output[idx_test], 1), labels[idx_test])

			if  loss_11.item() <= loss_min_1 :
				loss_min_1 = np.min((loss_11.item(), loss_min_1))
				bad_counter = 0
				best_acc_1 = model_one_test.item()
			else:
				bad_counter += 1

			if  loss_21.item() <= loss_min_2 :
				loss_min_2 = np.min((loss_21.item(), loss_min_2))
				bad_counter = 0
				best_acc_2 = model_two_test.item()
			else:
				bad_counter += 1

			best_acc = best_acc_2

			print("\r\rEpoch:%04d || " % epoch, "Train_acc_one:%5.4f%% || " % (train_acc1.item()*100), \
				 "Train_acc_two:%5.4f%% || " % (train_acc2.item()*100), \
				 "Best_acc:%5.4f%% || " % (best_acc*100), "Pseudo_labels:%04d || " % num, \
				 "Bad_counter:%02d" % bad_counter, end='')

			if bad_counter >= args.patience and t == args.t1 :
				bad_counter=0
				t = args.t2
			if bad_counter >= args.patience :
				break
		
	model.eval()
	model_one_test_output, model_two_test_output, loss_11, loss_21, loss_12, loss_22 = model(features, adj, idx_train, labels)
	model_one_testacc = accuracy(F.softmax(model_one_test_output[idx_test], 1), labels[idx_test])
	model_two_testacc = accuracy(F.softmax(model_two_test_output[idx_test], 1), labels[idx_test])
	
	return model_one_testacc, model_two_testacc, best_acc

def main():
	acc1 = []
	acc2 = []
	acc3 = []
	seed = random.randint(0,1000)
	for i in range(10):
		print("*******************************************************************************************************")
		#Loading data
		adj, features, labels, idx_train, idx_val, idx_test = load_data(args.dataset, args.public, args.percent, seed+i)
		print("The times:%d"%(i+1))
		print("Dataset:",args.dataset)
		print("Label_rate:",args.percent)
		print("idx_train:",list(idx_train.cpu().numpy()))
		features = RNM_filter(features, adj, args.k)  #RNM filter
		model_one_testacc, model_two_testacc, best_acc = Graph_pseudo(adj, features, labels, idx_train, idx_val, idx_test)
		print("\n",end='')
		print('The times:{} || '.format(i+1),'Model_one_testacc:{:.4f}% || '.format(model_one_testacc*100), \
		 'Model_two_testacc:{:.4f}% || '.format(model_two_testacc*100), 'Best_acc:{:.4f}% '.format(best_acc*100))
		acc1.append(model_one_testacc.item())
		acc2.append(model_two_testacc.item())
		acc3.append(best_acc)
		print('Test_acc:{:.2f}%'.format(best_acc*100))
		print("*******************************************************************************************************")
		print('\n',end='')
	mean_acc1 = np.mean(acc1)
	mean_acc2 = np.mean(acc2)
	mean_acc3 = np.mean(acc3)
	print("*******************************************************************************************************")
	print("Dataset:",args.dataset)
	if args.public == 0 :
		print("Label_rate:",args.percent)
	else :
		print("20 labels pre class.")
	print("Training set size:",len(idx_train))
	print("Validation set size:",len(idx_val))
	print("Test set size:",len(idx_test))
	print("Random seed:", seed)
	print("Use validation:",not args.fastmode)
	print("Acc_all:",list(acc3))
	print('Acc_mean:{:.2f}%'.format(mean_acc3*100))
	print("*******************************************************************************************************")
if __name__ == "__main__":
	main()
