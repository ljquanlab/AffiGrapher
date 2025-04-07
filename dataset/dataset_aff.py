
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
class ESDataset_aff(Dataset):

    def __init__(self, keys,args, data_dir,debug = False):
        super(ESDataset_aff, self).__init__()
        self.keys = keys
        self.data_dir = data_dir
        self.debug = debug
        self.args = args
        self.graphs = []
        self.error = []
 
        file_path = 'INDEX_general_NL.2020'


        if not args.test:
            # load data from LMDB database
            env = lmdb.open(args.lmdb_cache, map_size=int(1e8), max_dbs=1, readonly=True)
            self.graph_db = env.open_db('data'.encode()) # graph database
            self.txn = env.begin(buffers=True,write=False)
           
        else:
            pass
    
    def __len__(self):
        if self.debug:
            return 30000
        return len(self.keys)
    def collate(self, samples):

        """ 
        The input samples is a list of pairs (graph, label)
        collate function for building graph dataloader

        """
        samples = list(filter(lambda  x : x is not None,samples))
        g,full_g,key,Y = map(list, zip(*samples))
        batch_g = dgl.batch(g)
        batch_full_g = dgl.batch(full_g)
        Y = torch.tensor(Y)
        
        return batch_g,batch_full_g,Y,key
    def map_to_range(self, Y, minY, maxY):
        # 新区间的最小值和最大值
        newMin, newMax = 0, 5
        
        # 计算映射系数
        scale = (newMax - newMin) / (maxY - minY)
        
        # 应用映射规则
        newY = newMin + (Y - minY) * scale
        
        return newY
    def map_to_new_range(self,x, original_mean , original_std , new_min = 0, new_max = 5):
        # 计算标准化分数
        standardized = (x - original_mean) / original_std
        # 将标准化分数映射到新区间
        return new_min + (standardized * (new_max - new_min))
    def normalized(numbers):
        # 计算最小值和最大值
        min_value = min(numbers)
        max_value = max(numbers)

        # 归一化列表
        normalized_numbers = [10* (x - min_value) / (max_value - min_value) for x in numbers]
        return normalized_numbers

    def __getitem__(self, idx):
        key = self.keys[idx]
        # print(key)
        # rna = f"home/jkwang/RNA_dock_score/data/data/{key.split('/')[-2]}/rna/dockprep/rna.pdb"
        if not self.args.test:
            try:
                g,full_g,Y= pickle.loads(self.txn.get(key.encode(), db=self.graph_db))
                
            except Exception as e:
                print(f'file: {key} is not a valid file! Error: {e}')
               
        else:
            try:
                g, Y = self._GetGraph(rna, key, self.args)
            except:
                return None
        g.ndata['coors'] = full_g.ndata['coors']
        # Y = self.pdb_dict[key[0:4]]
        return g,full_g,key,Y


    @staticmethod
    def _GetGraph(self, rna_path, ligand_path, args):
        """
        construct structual graph based on covalent bond and non-bond interaction and save to LMDB database for speed up data loading
        Parameters
        ----------
        key : string 
            file path
        args : dictionary
            parameters 
	
		Returns
		-------
		(dgl.DGLGraph, Tensor)

        
        """
        # try:
        #     try:
        #         with open(key, 'rb') as f:
        #             m1,m2= pickle.load(f)
        #     except:
        #         with open(key, 'rb') as f:
        #             m1,m2,atompairs,iter_types= pickle.load(f)
        # except:
        #     return None
        m1 = Chem.MolFromPDBFile(rna_path, removeHs=True)
        # print(rna_path)
        # Check if the conversion was successful
        # if m1 is not None:
        #     print("Conversion successful.")
        #     # You can now manipulate or inspect the Mol object
        # else:
        #     print("Conversion failed.")
        # Create an SDMolSupplier object
        supplier = Chem.SDMolSupplier(ligand_path)
        # supplier = Chem.MolFromPDBFile(ligand_path, removeHs=True)
        
     
        # for mol in supplier:
        #     if mol is not None:
        #         print("Molecule loaded successfully.")
        #         # You can now manipulate or inspect the Mol object
        #     else:
        #         print("Failed to load a molecule.")
        # for m2 in supplier:
        m2 = supplier[0]
        n1,d1,adj1 = get_mol_info(m1)
        n2,d2,adj2 = get_mol_info(m2)

        from rdkit.Chem.rdchem import Mol
        # print(SplitMolByPDBResidues(m1).values())

        # for atom in m2.GetAtoms():
        #     atom.SetUnsignedProp("mapindex", atom.GetIdx())
        #     print(atom.GetIdx())

        H1 = np.concatenate([get_atom_graphformer_feature(m1,FP = False) ,np.array([0]).reshape(1,-1).repeat(n1,axis = 0)],axis=1)
        H2 = np.concatenate([get_atom_graphformer_feature(m2,FP = False) ,np.array([1]).reshape(1,-1).repeat(n2,axis = 0)],axis=1)
        # if args.virtual_aromatic_atom:
        if True:
            adj1,H1,d1,n1 = add_atom_to_mol(m1,adj1,H1,d1,n1)
            adj2,H2,d2,n2 = add_atom_to_mol(m2,adj2,H2,d2,n2)
        H = torch.from_numpy(np.concatenate([H1, H2], 0))
        # get covalent bond edges
        agg_adj1 = np.zeros((n1+n2, n1+n2))
        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2
        # get non-bond interactions based edges
        # if args.fingerprintEdge:

        if True:
            # slowly when trainging from raw data ,so we save the result in disk,for next time use
            if 'inter_types' not in vars().keys() and 'atompairs' not in vars().keys():
                try:
                    atompairs,iter_types = get_nonBond_pair(m2,m1)
                except:
                    atompairs,iter_types = [],[]
                    # print(key)
                # print("---1--")
                # with open("1AJU.pickle",'wb') as f:
                #     pickle.dump((m1,m2,atompairs,iter_types),f)
                # f.close()
            if len(atompairs) > 0:
                temp_fp= np.array(atompairs)
                u,v = list(temp_fp[:,0]) +  list((n1+ temp_fp[:,1])),list((n1+ temp_fp[:,1])) + list(temp_fp[:,0])
                agg_adj1[u,v] = 1

        agg_adj1 = torch.from_numpy(agg_adj1)
        adj_graph_1 = np.copy(agg_adj1)
        pocket = (m1,m2)
        item_1 = mol2graph(pocket,H,args,adj = adj_graph_1,n1 = n1,n2 = n2,\
            dm = (d1,d2) )
        g = preprocess_item(item_1, args,adj_graph_1)
        # print("---2--")
        g.ndata['coors'] = torch.from_numpy(np.concatenate([d1,d2],axis=0))
        valid = torch.zeros((n1+n2,))
        # set readout feature for task layer
        if args.pred_mode == 'ligand':
            valid[:n1] = 1
        elif args.pred_mode == 'protein':
            valid[n1:] = 1
        else:
            raise ValueError(f'not support this mode : {args.pred_mode} plz check the args')
        # path_parts = ligand_path.split(os.sep)
        # target_directory = "/public/home/qiang/jkwang/RNA_dock_score/data/Graph"
        # new_file_name = f"{path_parts[-6]}_{path_parts[-3]}_{path_parts[-2]}_{os.path.basename(ligand_path)}"
        # key = os.path.join(target_directory, new_file_name)
#             if args.loss_fn == 'mse_loss':

#                 Y = -float(key.split('-')[1].split('_')[0])
#             else:
#                 if '_active' in key.split('/')[-1]:
#                     Y = 1 
#                 else:
#                     Y = 0 
        Y = ligand_path.split('_')[-1]
        g.ndata['V'] = valid.float().reshape(-1,1)
        # path_parts = key.split(os.sep)
        # mid = f"{path_parts[-6]}_{path_parts[-3]}_{path_parts[-2]}_{os.path.basename(file)}"
        # with env.begin(write=True) as txn:
        #     txn.put(key.encode(), pickle.dumps((g,y)), db = dgl_graph_db)
        
        return g, Y
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



