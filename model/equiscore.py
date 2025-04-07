import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import WeightAndSum
import dgl.function as fn
import e3nn
from e3nn import o3
"""
with edge features
"""
import matplotlib.pyplot as plt
from model.equiscore_layer import EquiScoreLayer
from utils.equiscore_utils import MLPReadout
class conLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.05, verbose=False):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )    
            if self.verbose: print("Denominator", denominator)
                
            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
            return loss_ij.squeeze(0)
 
        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2*N) * loss
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.3)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)  # 在第一层后添加Dropout
        x = F.relu(self.fc2(x))
        x = self.drop(x)  # 在第二层后添加Dropout
        x = torch.sigmoid(self.fc3(x))
        return x

class CrossAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        self.out = nn.Linear(out_features, out_features)

    def forward(self, ligand, pocket):
        # 计算Query, Key, Value
        Q = self.query(ligand)
        K = self.key(pocket)
        V = self.value(pocket)

        # 计算注意力权重
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算输出特征
        attention_output = torch.matmul(attention_weights, V)
        output = self.out(attention_output)

        return output
class EquiScore(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        atom_dim = 16*12 if self.args.FP else 10*6
        self.atom_encoder = nn.Embedding(atom_dim  + 1, self.args.n_out_feature, padding_idx=0)
        self.edge_encoder = nn.Embedding( 36* 5 + 1, self.args.edge_dim, padding_idx=0) if args.edge_bias is True else nn.Identity()
        self.rel_pos_encoder = nn.Embedding(512, self.args.edge_dim, padding_idx=0) if args.rel_pos_bias is True else nn.Identity()#rel_pos
        self.in_degree_encoder = nn.Embedding(10, self.args.n_out_feature, padding_idx=0) if args.in_degree_bias is True else nn.Identity()
        # self.irreps_sh='1x0e+1x1e+1x2e'
        # self.irreps_edge_attr = o3.Irreps(irreps_sh)
        
        if args.rel_pos_bias:
            self.linear_rel_pos =  nn.Linear(self.args.edge_dim, self.args.head_size) 
        if self.args.lap_pos_enc:
            self.embedding_lap_pos_enc = nn.Linear(self.args.pos_enc_dim, self.args.n_out_feature)
        self.layers = nn.ModuleList([ EquiScoreLayer(self.args) \
                for _ in range(self.args.n_graph_layer) ]) 
        # for layer in self.layers[:2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        self.MLP_layer = MLPReadout(self.args)   # 1 out dim if regression problem   
        # self.weight_and_sum_kl = WeightAndSum(self.args.n_out_feature)     
        self.weight_and_sum = WeightAndSum(self.args.n_out_feature)   
        self.discriminator = Discriminator(input_dim=2 * self.args.n_out_feature) 
        # self.weight_and_sum2 = WeightAndSum(self.args.n_out_feature)     
        # self.cross_attention = CrossAttention(in_features=args.n_out_feature, out_features=args.n_out_feature)
    def getAtt(self,g,full_g):
        h = g.ndata['x']

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        for conv in self.layers:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer 
        # h = h * g.ndata['V']
        
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    def getAttFirstLayer(self,g,full_g):
        """
        A tool function to get the attention of the first layer
        """
        h = g.ndata['x']
        

        h = self.atom_encoder(h.long()).mean(-2)

        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)

        for conv in [self.layers[0]]:
            h, e = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
        # only ligand atom's features are used to task layer
        # h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        hg = self.MLP_layer(hg)
        
        return h,g,full_g,hg
    
    def forward(self, g, full_g, contrastive=False, getScore = False):
        """
        Parameters
        ----------
        g : dgl.DGLGraph 
            convalent and IFP based graph 

        full_g :dgl.DGLGraph
            geometric based graph

		Returns
		-------
            probability of binding

        """
        h = g.ndata['x']
        h = self.atom_encoder(h.long()).mean(-2)
        g.apply_edges(fn.u_sub_v('coors', 'coors', 'detla_coors'))
        h_vector = g.edata['detla_coors'].float()
        
        if self.args.lap_pos_enc:
            h_lap_pos_enc = g.ndata['lap_pos_enc']
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float()) 
            h = h + h_lap_pos_enc
        if self.args.in_degree_bias:
            h = h+ self.in_degree_encoder(g.ndata['in_degree'])
        e = self.edge_encoder(g.edata['edge_attr']).mean(-2)
        # if self.args.rel_pos_bias:
        #     if self.equi:
        #         edge_sh = o3.spherical_harmonics(l=self.irreps_edge_attr,x=h_vector, normalize=True, normalization='component')
        #         print(edge_sh.shape)
        #     else:
        #         e_bias = self.linear_rel_pos(h_vector)
        #     # print(h.shape)
        #     # print(e.shape)
        #     # print(e_bias.shape)
        #     e = e + e_bias
        i = 0
        for conv in self.layers:
            
            h, e, score_matrix = conv(g,full_g,h,e)
            h = F.dropout(h, p=self.args.dropout_rate, training=self.training)
            e = F.dropout(e, p=self.args.dropout_rate, training=self.training)
            # if contrastive and i == 1:  # 只在前两层使用对比学习
            #     # h_contrastive = h.clone()  # 保存前两层的嵌入用于对比学习
            #     h_contrastive = self.weight_and_sum_kl(g,h)
            #     # print(h_contrastive.shape)
            #     # print(h.shape)
            #     return h_contrastive
            # i+=1
        
        if self.args.crossatt:
            ligand = h * g.ndata['V']
            pocket = h * (1 - g.ndata['V'])
            # 计算ligand和pocket的表征
            ligand_repr = self.weight_and_sum(g, ligand)
            pocket_repr = self.weight_and_sum2(g, pocket)

            # 计算cross-attention
            cross_attn_output = self.cross_attention(pocket_repr, ligand_repr)

            # 计算亲和力值
            affinity = self.MLP_layer(cross_attn_output)
            return affinity
        if self.args.ligandonly:
            h = h * g.ndata['V']
        hg = self.weight_and_sum(g,h)
        if contrastive:
            return hg,self.MLP_layer(hg)
        if getScore:
            return self.MLP_layer(hg),score_matrix
        else: 
            return self.MLP_layer(hg)
