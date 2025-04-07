
import lmdb
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sys
sys.path.append("/public/home/qiang/jkwang/EquiScore-main")
import utils.utils as utils 
import numpy as np
import torch
import random
import math
from scipy.spatial import distance_matrix
import pickle
import dgl
import dgl.data
from utils.ifp_construct import get_nonBond_pair
from utils.dataset_utils import *
import pandas as pd
from torch_geometric.data import Data
random.seed(42)
from torch_cluster import radius_graph
class DTISampler(Sampler):
    """"
    weight based sampler for DTI dataset
    """
    def __init__(self, weights, num_samples, replacement=True):

        weights = np.array(weights)/np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    def __iter__(self):

        retval = np.random.choice(len(self.weights), self.num_samples, replace=self.replacement, p=self.weights) 
        return iter(retval.tolist())
    def __len__(self):
        return self.num_samples
def dgl_to_pyg(dgl_graph: dgl.DGLGraph) -> Data:
    # Extract node features
    # Assuming V, x, and in_degree are all part of the node features
    V = dgl_graph.ndata['V']
    x = dgl_graph.ndata['x']
    in_degree = dgl_graph.ndata['in_degree']
    
    # Concatenate them into a single node feature matrix
    node_attr = torch.cat([V, x, in_degree.view(-1, 1)], dim=-1)

    # Extract edge indices (PyG uses COO format: [2, num_edges])
    src, dst = dgl_graph.edges()
    edge_index = torch.stack([src, dst], dim=0)

    # Extract edge features (edge_attr in this case)
    edge_attr = dgl_graph.edata['edge_attr']

    # Convert to PyG Data object
    pyg_data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    
    return pyg_data, V
class ESDataset_contrastive(Dataset):

    def __init__(self, pos, neg, groups,args, data_dir,debug = False):
        super(ESDataset_contrastive, self).__init__()
        self.pos = pos
        self.neg = neg
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        self.error = []
        env = lmdb.open(args.lmdb_cache, map_size=int(1e8), max_dbs=1, readonly=True)
        self.graph_db = env.open_db('data'.encode()) # graph database
        self.txn = env.begin(buffers=True,write=False)
        self.groups = groups
        self.group_keys = list(groups.keys())
        self.pos_index = 0
        # print(groups)
    def __len__(self):
        if self.debug:
            return 30000
        # return 100*self.args.batch_size
        return len(self.group_keys)*self.args.batch_size
    def collate(self, samples):

        """ 
        The input samples is a list of pairs (graph, label)
        collate function for building graph dataloader

        """
        samples = list(filter(lambda  x : x is not None,samples))
        # g_pos1,full_g_pos1,key_pos,Y_pos1,g_pos2,full_g_pos2,Y_pos2, g_neg1,full_g_neg1,key_neg,Y_neg1,g_neg2,full_g_neg2,Y_neg2 = map(list, zip(*samples))
        # g_batch = dgl.batch([g_pos1, g_pos2, g_neg1, g_neg2]).to(args.device, non_blocking=True)
        # full_g_batch = dgl.batch([full_g_pos1, full_g_pos2, full_g_neg1, full_g_neg2]).to(args.device, non_blocking=True)
        g_pos1,full_g_pos1,key_pos,Y_pos1,g_pos2,full_g_pos2,Y_pos2, g_neg1,full_g_neg1,key_neg,Y_neg1,g_neg2,full_g_neg2,neg_pdbid2,Y_neg2 = map(list, zip(*samples))
        g_batch = dgl.batch(g_pos1 + g_pos2 + g_neg1 + g_neg2)
        full_g_batch = dgl.batch(full_g_pos1 + full_g_pos2 + full_g_neg1 + full_g_neg2)
        # 合并 Y 值
        Y_batch = Y_pos1 + Y_pos2 + Y_neg1 + Y_neg2
        Y_batch = torch.tensor(Y_batch)
        return g_batch,full_g_batch,Y_batch
#         batch_g_pos1 = dgl.batch(g_pos1)
#         batch_full_g_pos1 = dgl.batch(full_g_pos1)
#         batch_g_pos2 = dgl.batch(g_pos2)
#         batch_full_g_pos2 = dgl.batch(full_g_pos2)
#         batch_g_neg1 = dgl.batch(g_neg1)
#         batch_full_g_neg1 = dgl.batch(full_g_neg1)
#         batch_g_neg2 = dgl.batch(g_neg2)
#         batch_full_g_neg2 = dgl.batch(full_g_neg2)
#         # Y = torch.tensor(Y)
        
#         return batch_g_pos1,batch_full_g_pos1,batch_g_pos2,batch_full_g_pos2,batch_g_neg1,batch_full_g_neg1,batch_g_neg2,batch_full_g_neg2

    def __getitem__(self, idx):
        
        # key_pos = self.pos[idx]
        # key_neg = random.choice(self.neg)
        # print(key)
        # rna = f"home/jkwang/RNA_dock_score/data/data/{key.split('/')[-2]}/rna/dockprep/rna.pdb"
        def get_random_key_from_group(group):
            return random.choice(group)

        def get_two_random_keys_from_group(group):
            if len(group) < 2:
                return None, None
            key1, key2 = random.sample(group, 2)
            return key1, key2

        def get_random_group(groups):
            return random.choice(list(groups.values()))

        if not self.args.test:
            
            
            while True:
                # 随机选择一个 group
                # pdbid, group = random.choice(list(self.groups.items()))

                pdbid = self.group_keys[idx % len(self.group_keys)]
                group = self.groups[pdbid]
                oldpid = pdbid
                
                flag = True

                while len(group)<2:
                    flag = False
                    pdbid, group = random.choice(list(self.groups.items()))
                # 在该 group 中随机选择两个不同的 key 作为 pos1 和 pos2
                key_pos1, key_pos2 = get_two_random_keys_from_group(group)
                if key_pos1 is None or key_pos2 is None:
                    continue  # 如果 group 长度小于 2，则选择另一个 group

                # 随机选择另一个 group，并在其中随机选择一个 key 作为 neg1
                while True:
                    if flag:
                        neg_pdbid, neg_group = random.choice(list(self.groups.items()))
                    else:
                        neg_pdbid = oldpid
                        neg_group = self.groups[oldpid]
                        key_neg1 = get_random_key_from_group(neg_group)
                        key_neg2 = key_neg1
                        break
                    if neg_pdbid != pdbid:
                        key_neg1 = get_random_key_from_group(neg_group)
                        while True:
                            neg_pdbid2, neg_group2 = random.choice(list(self.groups.items()))
                            if neg_pdbid2 != pdbid and neg_pdbid2 != neg_pdbid:
                                key_neg2 = get_random_key_from_group(neg_group2)
                                break
                        break

                try:
                    # print(key_pos1)
                    g_pos1, full_g_pos1, Y_pos1 = pickle.loads(self.txn.get(key_pos1.encode(), db=self.graph_db))
                    g_pos2, full_g_pos2, Y_pos2 = pickle.loads(self.txn.get(key_pos2.encode(), db=self.graph_db))
                    g_neg1, full_g_neg1, Y_neg1 = pickle.loads(self.txn.get(key_neg1.encode(), db=self.graph_db))
                    g_neg2, full_g_neg2, Y_neg2 = pickle.loads(self.txn.get(key_neg2.encode(), db=self.graph_db))
                    break  # 成功加载数据后退出循环
                except Exception as e:
                    print(f'Error loading data for keys: {key_pos1}, {key_pos2}, {key_neg1}. Error: {e}')
        g_pos1.ndata['coors'] = full_g_pos1.ndata['coors']
        g_pos2.ndata['coors'] = full_g_pos2.ndata['coors']
        g_neg1.ndata['coors'] = full_g_neg1.ndata['coors']
        g_neg2.ndata['coors'] = full_g_neg2.ndata['coors']
        return g_pos1,full_g_pos1,pdbid,Y_pos1,g_pos2,full_g_pos2,Y_pos2, g_neg1,full_g_neg1,neg_pdbid,Y_neg1,g_neg2,full_g_neg2,neg_pdbid2,Y_neg2
        
        
        if not self.args.test:
            try:
                g_pos1,full_g_pos1,Y_pos1 = pickle.loads(self.txn.get(key_pos[0].encode(), db=self.graph_db))
                g_neg1,full_g_neg1,Y_neg1 = pickle.loads(self.txn.get(key_neg[0].encode(), db=self.graph_db))
                g_pos2,full_g_pos2,Y_pos2 = pickle.loads(self.txn.get(key_pos[1].encode(), db=self.graph_db))
                g_neg2,full_g_neg2,Y_neg2 = pickle.loads(self.txn.get(key_neg[1].encode(), db=self.graph_db))
            except Exception as e:
                print(f'file: {key_pos} is not a valid file! Error: {e}')
                # print(key)
                # while True:
                #     try:
                #         idx = random.randint(0, len(self.keys) - 1)
                #         key = self.keys[idx]
                #         g, Y= pickle.loads(self.txn.get(key.encode(), db=self.graph_db))
                #         break
                #     except:
                #         print(key)
                # idx = random.rand
                # key = self.keys[idx]
                # g,Y = self._GetGraph(rna,key,self.args)
        else:
            try:
                g, Y = self._GetGraph(rna, key, self.args)
            except:
                return None
        g_pos1.ndata['coors'] = full_g_pos1.ndata['coors']
        g_pos2.ndata['coors'] = full_g_pos2.ndata['coors']
        g_neg1.ndata['coors'] = full_g_neg1.ndata['coors']
        g_neg2.ndata['coors'] = full_g_neg2.ndata['coors']
        return g_pos1,full_g_pos1,key_pos,Y_pos1,g_pos2,full_g_pos2,Y_pos2, g_neg1,full_g_neg1,key_neg,Y_neg1,g_neg2,full_g_neg2,Y_neg2
        

 
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.parsing import parse_train_args


if __name__ == "__main__":
    args = parse_train_args()
    with open ('trainData.pkl', 'rb') as fp:
        train_keys = list(pickle.load(fp))
    # print(len(train_keys))
    train_dataset = ESDataset_pyg(train_keys,args, args.data_path,args.debug)#keys,args, data_dir,debug
    
    for i, data in enumerate(train_dataset):
        print(f"Item {i}: {data}")
        break



