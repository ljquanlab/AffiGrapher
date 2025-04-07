import torch
import numpy as np
from utils import *
from rdkit import Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
import rdkit
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import dgl
import os.path as osp
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
# ===================== BOND START =====================
possible_bond_type_list = list(range(32))
possible_bond_stereo_list = list(range(16))
possible_is_conjugated_list = [False, True]
possible_is_in_ring_list = [False, True]
possible_bond_dir_list = list(range(16))
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
def bond_to_feature_vector(bond):
    """
    input: rdkit.Chem.rdchem.Bond
    output: bond_feature (list)
    
    """
    # 0
    bond_type = int(bond.GetBondType())
    assert bond_type in possible_bond_type_list

    bond_stereo = int(bond.GetStereo())
    assert bond_stereo in possible_bond_stereo_list

    is_conjugated = bond.GetIsConjugated()
    assert is_conjugated in possible_is_conjugated_list
    is_conjugated = possible_is_conjugated_list.index(is_conjugated)

    is_in_ring = bond.IsInRing()
    assert is_in_ring in possible_is_in_ring_list
    is_in_ring = possible_is_in_ring_list.index(is_in_ring)

    bond_dir = int(bond.GetBondDir())
    assert bond_dir in possible_bond_dir_list

    bond_feature = [
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        bond_dir,
    ]
    return bond_feature

def GetNum(x,allowed_set):
    """
    input: 
        x (int)
        allowed_set (list)
    output:    
        [index] (list)
    
    """
    try:
        return [allowed_set.index(x)]
    except:
        return [len(allowed_set) -1]
def get_aromatic_rings(mol:rdkit.Chem.Mol) -> list:

    """
    input: 
        rdkit.Chem.Mol
    output:
       aromaticatoms rings (list)
    """
    Chem.SanitizeMol(mol, Chem.SANITIZE_ALL)
    aromaticity_atom_id_set = set()
    rings = []
    for atom in mol.GetAromaticAtoms():
        aromaticity_atom_id_set.add(atom.GetIdx())

    # print("Aromatic atoms:", [atom.GetIdx() for atom in mol.GetAromaticAtoms()])
    # get ring info 
    ssr = Chem.GetSymmSSSR(mol)
    # print(ssr)
    for ring in ssr:
        # print("Ring:", list(ring))  # 打印环信息（调试用）
        ring_id_set = set(ring)
        # check atom in this ring is aromaticity
        if ring_id_set <= aromaticity_atom_id_set:
            rings.append(list(ring))
    return rings
def add_atom_to_mol(mol:rdkit.Chem.Mol,adj:np.array,H:np.array,d:np.array,n:int, bases, offset=0,):
    """
    docstring: 
        add virtual aromatic atom feature/adj/3d_positions to raw data
    input:
        mol: rdkit.Chem.Mol
        adj: adj matrix
        H: node feature
        node d: 3d positions
        n: node nums 
    """
    assert len(adj) == len(H),'adj nums not equal to nodes'
    # print(111111)
    rings = get_aromatic_rings(mol)
    num_aromatic = len(rings)
    h,b = adj.shape
    all_zeros = np.zeros((num_aromatic+h,num_aromatic+b))

    mapping = {}   # 初始化  节点与原子编号映射
    #add all zeros vector to bottom and right
    all_zeros[:h,:b] = adj
    # print(1111)
    # print(rings)
    for i,ring in enumerate(rings):
        all_zeros[h+i,:][ring] = 1
        all_zeros[:,h+i][ring] = 1
        all_zeros[h+i,:][h+i] = 1
        d = np.concatenate([d,np.mean(d[ring],axis = 0,keepdims=True)],axis = 0)
        H  = np.concatenate([H,np.array([15]*(H.shape[1]))[np.newaxis]],axis = 0)
        base_info = [bases[atom] for atom in ring]
        mapping[h + i + offset] = {'type': 'aromatic','rdkit_index': h + i + offset,'type_id': 11, 'atoms': ring, 'bases': base_info[0]}
        # print(mapping[h + i + offset])
    for atom_idx in range(n):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_type = GetNum(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H','other'])
        mapping[atom_idx + offset] = {'type': 'atom', 'type_id': atom_type[0], 'rdkit_index': atom_idx, 'bases': bases[atom_idx]}
    assert len(all_zeros) == len(H),'adj nums not equal to nodes'
    return all_zeros,H,d,n+num_aromatic,mapping
def get_mol_info(m1, isRNA = True):
    """
    input: 
        rdkit.Chem.Mol
    output:
        n1: node nums
        d1: 3d positions
        adj1: adj matrix
    """
    # print(type(m1))
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(m1)+np.eye(n1)
    # 获取每个原子的碱基信息
    bases = []
    if isRNA:
        for atom in m1.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info is not None:
                base_name = monomer_info.GetResidueName()  # 碱基名称，如 A, G, C
                base_number = monomer_info.GetResidueNumber()  # 碱基编号，如 1, 2, 3
                base_id = f"{base_name}_{base_number}"  # 如 A1, G2
            else:
                base_id = "Unknown"  # 如果无法获取，标记为 Unknown
            bases.append(base_id)
    else:
        for atom in m1.GetAtoms():
            base_id = 'ligand_1'
            bases.append(base_id)

    return n1, d1, adj1, bases
    return n1,d1,adj1
def atom_feature_graphformer(m, atom_i, i_donor, i_acceptor):
    """
    docstring:
        atom feature as same as graphformer
    input:
        m: rdkit.Chem.Mol
        atom_i: atom index
        i_donor: H donor or not 
        i_acceptor: H acceptor or not 
    output:
        atom feature (list)
    """
    atom = m.GetAtomWithIdx(atom_i)
    # print(atom)
    return np.array(GetNum(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    GetNum(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    GetNum(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    GetNum(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28
#   ['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other']) \
def atom_feature_attentive_FP(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    """
    docstring:
        atom feature as same as attentiveFP
    input:
        m: rdkit.Chem.Mol
        atom_i: atom index
        i_donor: H donor or not 
        i_acceptor: H acceptor or not 
    output:
        atom feature (array)
    """
                  
    if bool_id_feat:
        return np.array([atom_to_id(atom)])
    else:
        results = GetNum(
          atom.GetSymbol(),
          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H','other']) \
              + GetNum(atom.GetDegree(),[0, 1, 2, 3, 4, 5]) + \
                  [int(atom.GetFormalCharge()), int(atom.GetNumRadicalElectrons())] + \
                  GetNum(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]

        if not explicit_H:
            results = results + GetNum(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + GetNum(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)
def get_atom_graphformer_feature(m,FP = False):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        if FP:
            H.append(atom_feature_attentive_FP(m.GetAtomWithIdx(i),
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True))
            # print(H[i])
        else:
            H.append(atom_feature_graphformer(m, i, None, None))
            # print(H[i])
    # print(len(H))
    # print(type(H[0]))
    H = np.array(H)        

    return H      

# ===================== BOND END =====================
def convert_to_single_emb(x, offset=35):
    """
    docstring:
        merge multiple embeddings into one embedding

    """
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

#===================3d position start ========================
def get_rel_pos(mol):
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        energy = res[index][1]
        conf = new_mol.GetConformer(id=int(index))
    except:
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
        conf = new_mol.GetConformer()

    atom_poses = []
    for i, atom in enumerate(new_mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(new_mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    atom_poses = np.array(atom_poses, dtype=float)
    rel_pos_3d = cdist(atom_poses, atom_poses)
    return rel_pos_3d
#===================3d position end ========================
#===================data attributes start ========================
# from dataset import *
def molEdge(mol,n1,n2,adj_mol = None):
    """
    docstring:
        get edges and edge features of mol
    input:
        mol: rdkit.Chem.Mol
        n1: number of atoms in mol
        # n2: number of atoms in adj_mol
        adj_mol: adjacent matrix of mol
    output:
        edges_list: list of edges
        edge_features_list: list of edge features
    """
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() 
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
    # add virtual aromatic nodes feature
    if adj_mol is None:
        return edges_list ,edge_features_list
    else:
        n = len(mol.GetAtoms())
        adj_mol -= np.eye(len(adj_mol))
        dm = adj_mol[n:n1,:n1]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v),list( edge_pos_v) + list((edge_pos_u + n))
        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))

    return edges_list ,edge_features_list
def pocketEdge(mol,n1,n2,adj_pocket = None):
    """"
    docstring:
        get edges and edge features of pocket
    input:
        mol: rdkit.Chem.Mol
        n1: number of atoms in mol
        # n2: number of atoms in adj_pocket
        adj_pocket: adjacent matrix of pocket
    output:
        edges_list: list of edges
        edge_features_list: list of edge features 
    """
    edges_list = []
    edge_features_list = []
    if len(mol.GetBonds()) > 0: # mol has bonds
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx() + n1
            j = bond.GetEndAtomIdx() + n1
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
    # add self edge feature
    if adj_pocket is  None :
        return edges_list ,edge_features_list
    # add virtual aromatic nodes feature
    # add pocket
    else:
        n = len(mol.GetAtoms()) + n1
        all_n = len(adj_pocket)
        adj_pocket -= np.eye(all_n)
        dm = adj_pocket[n:,n1:]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        
        u,v = list((edge_pos_u + n)) +  list( edge_pos_v + n1),list( edge_pos_v + n1) + list((edge_pos_u + n))

        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[33,17,3,3,17]]*len(u))
   
    return edges_list ,edge_features_list
def getEdge(mols,n1,n2,adj_in = None):
    """
    Docstring:
        merge molEdge and pocketEdge
    input:
        mols: list of rdkit.Chem.Mol
        n1: number of atoms in mol
        n2: number of atoms in pocket
        adj_in: adjacent matrix of mol and pocket
    output:
        edges_list: list of edges
        edge_features_list: list of edge features
    """
    num_bond_features = 5
    mol,pocket = mols
    mol1_edge_idxs,mol1_edge_attr = molEdge(mol,n1,n2,adj_mol = adj_in)
    mol2_edge_idxs,mol2_edge_attr = pocketEdge(pocket,n1,n2,adj_pocket = adj_in)
    edges_list = mol1_edge_idxs + mol2_edge_idxs
    edge_features_list = mol1_edge_attr + mol2_edge_attr
    # add self edge
    u,v = np.where(np.eye(n1+n2) == 1)
    edges_list.extend([*zip(u,v)])
    edge_features_list.extend([[34,17,4,4,18]]*len(u))
    if adj_in is  None:
        pass
    else:
        #add virtual edge , add fingerprint edges features
        dm = adj_in[:n1,n1:]
        edge_pos_u,edge_pos_v = np.where(dm == 1)
        
        u,v = list(edge_pos_u) +  list((n1+ edge_pos_v)),list((n1+ edge_pos_v)) + list(edge_pos_u)

        edges_list.extend([*zip(u,v)])
        edge_features_list.extend([[32,16,2,2,16]]*len(u))
    if len(edges_list) == 0:
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)
    else:
        edge_index = np.array(edges_list, dtype = np.int64).T
        edge_attr = torch.tensor(edge_features_list, dtype = torch.int64)
    return edge_index,edge_attr


def mol2graph(mol,x,args,n1,n2,adj = None,dm = None):
    """
    dcostring:
        Converts mol to graph Data object
    input: 
        mol: rdkit.Chem.Mol
        x: node features
        args: args
        n1: number of atoms in mol
        n2: number of atoms in pocket
        adj: adjacent matrix of mol and pocket
        dm: distance matrix of mol and pocket
    output: 
        graph object
    """
    if args.edge_bias:
        edge_index, edge_attr= getEdge(mol,adj_in = adj,n1 = n1,n2 = n2)
    else:
        edge_index, edge_attr= None,None

    if args.rel_3d_pos_bias and dm is not None:
        if len(dm) == 2:
            d1,d2 = dm
            rel_pos_3d = distance_matrix(np.concatenate([d1,d2],axis=0),np.concatenate([d1,d2],axis=0))
        else:
            rel_pos_3d = np.zeros((n1+n2, n1+n2))
            rel_pos_3d[:n1,n1:] = np.copy(dm)
            rel_pos_3d[n1:,:n1] = np.copy(np.transpose(dm))
    else:
        rel_pos_3d =  None
    graph = dict()

    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['rel_pos_3d'] = rel_pos_3d
    return graph 
#=================== data attrs end ========================
#===================all data attrs process start ========================
import pandas as pd
import numpy as np
import scipy.sparse as sp
def get_pos_lp_encoding(adj,pos_enc_dim = 8):
    """
    dcostring:
        get position laplace embedding
    input: 
        adj: adjacent
        pos_enc_dim: position embedding dim
    output: 
        position embedding (torch.tensor)
    """
    A = sp.coo_matrix(adj)
    N = sp.diags(adj.sum(axis = 1).clip(1) ** -0.5, dtype=float)
    L = sp.eye(len(adj)) - N * A * N
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    return lap_pos_enc
def pandas_bins(dis_matrix,num_bins = None,noise = False):
    """
    dcostring:
        Coarse-grained processing distance matrix
    input: 
        dis_matrix: distance matrix
        num_bins: number of bins
    output: 
        bins_index: bins index
    """
    if num_bins is None:
        num_bins = int((5-2.0)/0.05 + 1)
    if noise:
        t = np.random.laplace(0.001, 0.05)
        dis_matrix += t
    bins = [-1.0] + list(np.linspace(2.0,5,num_bins)) + [10000]
    shape = dis_matrix.shape
    bins_index = np.array(pd.cut(dis_matrix.flatten(),bins = bins,labels = [i for i in range(len(bins) -1)])).reshape(shape)
    return bins_index
def preprocess_item(item, args,adj,n1,n2):
    """
    dcostring:
        compute the edge attributions and node degree
    input: 
        item: graph will be processed
        adj: adjacent
    output: 
        g: processed graph
    """
    edge_attr, edge_index, x  = item['edge_feat'], item['edge_index'], item['node_feat']
    N = x.size(0)
    if args.model == 'EquiScore':
        offset = 16 if args.FP else 10
        x = convert_to_single_emb(x,offset = offset)
    if x.min()< 0:
        print('convert feat',x.min())
    
    adj = torch.tensor(adj,dtype=torch.long)
    # edge feature here
    g = dgl.graph((edge_index[0, :], edge_index[1, :]),num_nodes=len(adj))
    if args.lap_pos_enc:
        g.ndata['lap_pos_enc'] = get_pos_lp_encoding(adj.numpy(),pos_enc_dim = args.pos_enc_dim)
    g.ndata['x']  = x
    adj_in = adj.long().sum(dim=1).view(-1)
    adj_in = torch.where(adj_in < 0,0,adj_in)
    g.ndata['in_degree'] = torch.where(adj_in > 8,9,adj_in) if args.in_degree_bias else None
    g.edata['edge_attr'] = convert_to_single_emb(edge_attr)   
    
    return g

def preprocess_item_map(item, args, adj, n1, n2, ligand_mapping,rna_mapping ):
    """
    Preprocess the graph item by computing node features, edge features, and adding mappings
    for RNA and ligand atoms.
    """
    edge_attr, edge_index, x  = item['edge_feat'], item['edge_index'], item['node_feat']
    N = x.size(0)
    
    # Model-specific adjustments
    offset = 16 if args.FP else 10
    if args.model == 'EquiScore':
        x = convert_to_single_emb(x, offset=offset)
        
    if x.min() < 0:
        print('convert feat', x.min())
    
    adj = torch.tensor(adj, dtype=torch.long)
    
    # Create the DGL graph
    g = dgl.graph((edge_index[0, :], edge_index[1, :]), num_nodes=len(adj))
    
    # Add Laplacian position encoding if specified
    if args.lap_pos_enc:
        g.ndata['lap_pos_enc'] = get_pos_lp_encoding(adj.numpy(), pos_enc_dim=args.pos_enc_dim)
    
    # Node features
    g.ndata['x'] = x
    
    # Degree and edge attributes
    adj_in = adj.long().sum(dim=1).view(-1)
    adj_in = torch.where(adj_in < 0, 0, adj_in)
    g.ndata['in_degree'] = torch.where(adj_in > 8, 9, adj_in) if args.in_degree_bias else None
    
    g.edata['edge_attr'] = convert_to_single_emb(edge_attr)
    
    # Adding RNA and ligand mappings to node features
    # Initialize the node labels as an empty tensor (assuming 'type' and 'base' will be added)
    node_types = {}  # Example type
    node_idx = {}  # Example base info
    node_bases = {} 
    node_atom_idx = {}
    node_atom_type = {}
    # import re

    # base_pattern = re.compile(r"([A-Z]+)(\d+)")  # 匹配大写字母 + 数字
    base_category_map = {
    'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U',
    'CA': 'A', 'CU': 'U', 'CG': 'G', 'CC': 'C',
    'GA': 'A', 'GC': 'C', 'GG': 'G', 'GU': 'U',
    'DA': 'A', 'DU': 'U', 'DC': 'C', 'DG': 'G', 'T':'T', 'DT':'T','N':'N'
    }
    main_base_to_int = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4, 'N' : 5, 'Unknown': -1
    }
    # print("3")
    # Iterate through RNA nodes and assign corresponding information
    # print(len(rna_mapping))
    # print(len(ligand_mapping))
    for idx, atom_info in rna_mapping.items():
        node_types[idx] = 1  # Example: Mark as RNA atom (type = 1)
        # print(atom_info)
        # 使用正则表达式分割
        # base_name = "unknown"
        # match = base_pattern.match(atom_info['bases'])
        # print(atom_info['bases'].split('_'))
        # print(match)
        # if match:
        # print(atom_info['bases'])
        if atom_info['type']!="atom":
            node_types[idx] = 0 #aromatic type = 0
            # print(atom_info)
        base_name = atom_info['bases'].split('_')[0].strip()  # 提取碱基部分，例如 "CD"
        # print(base_name)
        node_id = int(atom_info['bases'].split('_')[1])  # 提取节点索引部分，例如 5
        # print(f"Base Name: {base_name}, Node Index: {node_id}")
        node_idx[idx] = int(node_id) # Example: store base information  like A5
        node_bases[idx] = main_base_to_int[base_category_map.get(base_name)] 
        node_atom_idx[idx] = atom_info['rdkit_index']
        node_atom_type[idx] = atom_info['type_id']
        
    # print("4")
    # Iterate through Ligand nodes and assign corresponding information
    for idx, atom_info in ligand_mapping.items():
        node_types[idx+len(rna_mapping)] = 2  # Example: Mark as Ligand atom (type = 2)
        # print(atom_info)
        base_name = atom_info['bases'].split('_')[0]  # 提取碱基部分，例如 "CD"
        node_id = int(atom_info['bases'].split('_')[1])  # 提取节点索引部分，例如 5
        # print(f"Base Name: {base_name}, Node Index: {node_id}")
        node_idx[idx+len(rna_mapping)] = 0 # Example: store base information ligand is zero
        node_bases[idx+len(rna_mapping)] = 5
        node_atom_idx[idx+len(rna_mapping)] = atom_info['rdkit_index'] + len(rna_mapping)
        # print(node_atom_idx[idx+len(rna_mapping)])
        node_atom_type[idx+len(rna_mapping)] = atom_info['type_id']
    # print("5")
    # Add 'type' and 'bases' as node features
    N = g.num_nodes()  # Total number of nodes in the graph
    node_types_tensor = torch.zeros(N, dtype=torch.int64)
    node_bases_tensor = torch.zeros(N, dtype=torch.int64)
    node_base_idx_tensor = torch.zeros(N, dtype=torch.int64)
    node_atom_idx_tensor = torch.zeros(N, dtype=torch.int64)
    node_atom_type_tensor = torch.zeros(N, dtype=torch.int64)

    # Fill the tensors using the dictionary values
    for idx, type_value in node_types.items():
        node_types_tensor[idx] = type_value

    for idx, base_value in node_bases.items():
        node_bases_tensor[idx] = base_value

    for idx, idx_value in node_idx.items():
        node_base_idx_tensor[idx] = idx_value

    for idx, idx_value in node_atom_idx.items():
        node_atom_idx_tensor[idx] = idx_value

    for idx, idx_value in node_atom_type.items():
        node_atom_type_tensor[idx] = idx_value

    # Assign the tensors to the graph's node data
    g.ndata['type'] = node_types_tensor  # rna :1 or ligand: 2 or virtual: 0
    g.ndata['bases'] = node_bases_tensor #base type in main_base_to_int;  ligand is 5 
    g.ndata['base_idx'] = node_base_idx_tensor #base_idx   liake G_1;A_2
    g.ndata['atom_idx'] = node_atom_idx_tensor #atom_idx 
    g.ndata['atom_type'] = node_atom_type_tensor #atom_type
    # print(node_node_idx_tensor)
    # print(g.ndata)
    return g

