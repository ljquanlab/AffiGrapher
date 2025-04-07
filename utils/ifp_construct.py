
import prolif as plf
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import traceback
import numpy as np
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
def get_nonBond_pair(m1,m2):
    """
    docstring: 
        calculate IFP(interaction fingerprint)
    input:
        m1: rdkit.Chem.Mol(ligand)
        m2: rdkit.Chem.Mol(protein)
    output:
        nonbond_pairs: list of tuple (ligand_atom_idx,protein_atom_idx)
        inter_types: list of interaction type
    """
    # print("pair")
    fp = plf.Fingerprint()
    prot = plf.Molecule(m2)
    # print("pair")
    ligand = plf.Molecule.from_rdkit(m1)
    # print("pair")
    fp.run_from_iterable([ligand],prot,progress=False,n_jobs = 1)
    df = fp.to_dataframe(return_atoms=True)
    # print("plf")
    # print(df)
    # find IFP  atom pairs , match with rdkit id 
    res_to_idx = defaultdict(dict)
    # 获取RNA和分子中的所有原子坐标
    rna_coords = m2.GetConformer().GetPositions()
    ligand_coords = m1.GetConformer().GetPositions()
    # 定义pocket的半径
    pocket_radius = 10
    pocket = []
    # 遍历RNA中的每个原子
    res_name = ''
    try:
        for atom in m2.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_pos = rna_coords[atom_idx]
            if res_name == str(plf.residue.ResidueId.from_atom(atom)):
                continue
            else:
                res_name = ''
            # 遍历分子中的每个原子
            for ligand_atom in m1.GetAtoms():
                ligand_atom_idx = ligand_atom.GetIdx()
                ligand_atom_pos = ligand_coords[ligand_atom_idx]

                # 计算RNA原子和分子原子之间的距离
                distance = np.linalg.norm(atom_pos - ligand_atom_pos)

                # 如果距离小于定义的半径，将该RNA原子加入到pocket中
                if distance <= pocket_radius:
                    res_name = str(plf.residue.ResidueId.from_atom(atom))
                    pocket.append(res_name)
    except Exception as e:
        print('pocket error')
        print(traceback.format_exc())
    # print(list(set(pocket)))            
    pocket = list(set(pocket)) 
    
    # print(res_to_idx)            
    for atom in m2.GetAtoms():
        atom_idx = atom.GetIdx()
        res_name = str(plf.residue.ResidueId.from_atom(atom))
        # print(res_name,atom_idx)
        res_to_idx[res_name][len(res_to_idx[res_name])] = atom_idx
    # print(len(m2.GetAtoms()))
    # print("pair2")
    nonbond_pairs = []
    inter_types = []
    subatom = []
    for key in df.keys():
        # print(key)
        ligand_name,res_name,inter_type = key
        lig_atom_num,res_atom_num = df[key][0]
        # print(df[key][0])
        pdb_atom_idx = res_to_idx[res_name][res_atom_num]
        nonbond_pairs.append((lig_atom_num,pdb_atom_idx))  
        for i in range(len(res_to_idx[res_name])):
            subatom.append(res_to_idx[res_name][i])
        inter_types.append(inter_type)
    for res_name in pocket:
        for i in range(len(res_to_idx[res_name])):
            subatom.append(res_to_idx[res_name][i])
    # print(subatom)
    subatom = list(set(subatom))
    return nonbond_pairs,inter_types,subatom