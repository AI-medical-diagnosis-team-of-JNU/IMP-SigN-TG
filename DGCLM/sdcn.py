from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from K_means_new import K_means
from collections import Counter





torch.cuda.set_device(1)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()

        self.enc_1 = nn.Conv2d(1, n_enc_1, 1, padding=0)
        self.enc_2 = nn.Conv2d(n_enc_1, n_enc_2, 1, padding=0)
        self.enc_3 = nn.Conv2d(n_enc_2, n_enc_3, 1, padding=0)
        self.z_layer = Linear(n_enc_3*n_input, n_z)

        self.enc_bn_1 = nn.BatchNorm2d(n_enc_1)
        self.enc_bn_2 = nn.BatchNorm2d(n_enc_2)
        self.enc_bn_3 = nn.BatchNorm2d(n_enc_3)

        self.dec_1 = nn.Conv2d(n_z, n_dec_1, 1, padding=0)
        self.dec_2 = nn.Conv2d(n_dec_1, n_dec_2, 1, padding=0)
        self.dec_3 = nn.Conv2d(n_dec_2, n_dec_3, 1, padding=0)

        self.dec_bn_1 = nn.BatchNorm2d(n_dec_1)
        self.dec_bn_2 = nn.BatchNorm2d(n_dec_2)
        self.dec_bn_3 = nn.BatchNorm2d(n_dec_3)

        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        x = x.view(-1, 1, 3, 1)

        enc_h1 = F.relu(self.enc_bn_1(self.enc_1(x)))
        enc_h2 = F.relu(self.enc_bn_2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.enc_bn_3(self.enc_3(enc_h2)))
        enc_h3 = enc_h3.view(x.shape[0],-1)
        z = self.z_layer(enc_h3)
        z = z.view(x.shape[0],-1,1,1)

        dec_h1 = F.relu(self.dec_bn_1(self.dec_1(z)))
        dec_h2 = F.relu(self.dec_bn_2(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.dec_bn_3(self.dec_3(dec_h2)))
        dec_h3 = dec_h3.view(x.shape[0],-1)

        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z.view(x.shape[0],-1)


class DGCLM(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(DGCLM, self).__init__()

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            )

        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1*n_input)
        self.gnn_2 = GNNLayer(n_enc_1*n_input, n_enc_2*n_input)
        self.gnn_3 = GNNLayer(n_enc_2*n_input, n_enc_3*n_input)
        self.gnn_4 = GNNLayer(n_enc_3*n_input, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)
        tra1 = tra1.view(x.shape[0], -1)
        tra2 = tra2.view(x.shape[0], -1)
        tra3 = tra3.view(x.shape[0], -1)
        # GCN Module

        h = self.gnn_1(x, adj)

        h = self.gnn_2(h+tra1, adj)
        h = self.gnn_3(h+tra2, adj)
        h = self.gnn_4(h+tra3, adj)

        h = self.gnn_5(h+z, adj, active=False)

        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_DGCLM(dataset):

    data = torch.Tensor(dataset.x).to(device)

    pretain_ae = AE(32,64,128,128,64,32,n_input=args.n_input,n_z=args.n_z).cuda()
    pretain_ae.load_state_dict(torch.load(args.pretrain_path, map_location='cuda'))
    with torch.no_grad():
        _, _, _, _, z = pretain_ae(data)

    k_model = K_means(z.data.cpu().numpy())
    best_k, best_kmeans = k_model.best_k()
    print("best_k:........................:", best_k)

    model = DGCLM(32, 64, 128, 128, 64, 32,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=best_k,
                 v=1.0).to(device)

    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, 6)
    adj = adj.cuda()
    print(adj.shape)

    # cluster parameter initiate

    with torch.no_grad():
        _, _, _, _, z = model.ae(data)

    model.cluster_layer.data = torch.tensor(best_kmeans.cluster_centers_).to(device)
    # eva(y, y_pred, 'pae')

    for epoch in range(20000):
        print(epoch)
        if epoch % 1 == 0:

            # update_interval
            print(data.shape, adj.shape)
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        x_bar, q, pred, _ = model(data, adj)

        # print(pred)

        pred_ = pred.data.cpu().numpy()

        f = "cluster_output.txt"
        with open(f,"r+") as file:
            file.truncate()
        for i in range(0,pred_.shape[0]):
            # print(pred[i])
            maxIndex = 0
            max = pred_[i][0]
            for j in range(1,pred_.shape[1]):
                if pred_[i][j] > max:
                    max = pred_[i][j]
                    maxIndex = j
            # print("epoch[",epoch,"], maxIndex:", str(maxIndex))
            with open(f,"a") as file:
                file.write(str(maxIndex)+"\n")
        if epoch == 19999:
            np.savetxt("./out.txt", pred_)
        # print(str(epoch)+":......................write finished")

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='feature')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--n_input',type=int,default=3)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)
'''
    if args.name == 'usps':
        args.n_clusters = 10
        args.n_input = 256

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334

    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
'''

print(args)
train_DGCLM(dataset)
