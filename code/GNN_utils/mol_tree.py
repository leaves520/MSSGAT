import pandas as pd
import copy,os
import numpy as np
from dgl import DGLGraph
import rdkit.Chem as Chem
from tqdm import tqdm

from .chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles


def get_Vocab(data_path = None): #对使用树分解后的分子的官能团簇写入词汇表 vocabulary.txt
    assert data_path is not None ,"Path to data must be required,plz check"
    assert isinstance(data_path,str), "Path to data must be string"

    if not os.path.exists(data_path):
        raise ValueError("Path to data is not found")

    # data_name = os.path.split(data_path.strip(".csv"))[-1]
    data_name = os.path.split(data_path)[-1].split('.')[0]
    output_path = './dataset' + '/vocabulary_' + data_name + '.txt'
    data=pd.read_csv(data_path)['Smiles']

    print('To get the vocabulary of {}'.format(data_name))
    result=set(())
    for i,smiles in enumerate(tqdm(data)):
        temp=DGLMolTree(smiles)
        for key in temp.nodes_dict:
            result.update([temp.nodes_dict[key]['smiles']])

    print('\n\tGet Vocabulary Finished!')

    with open(output_path,"w") as f:
        for csmiles in result:
            f.writelines(csmiles+"\n")


def get_Vocab_df(data_name,df,output_path = None): #对使用树分解后的分子的官能团簇写入词汇表 vocabulary.txt

    if output_path is None:
        output_path = './dataset' + '/vocabulary_' + data_name + '.txt'

    data=df['Smiles']

    print('To get the vocabulary of {}'.format(data_name))
    result=set(())
    for i,smiles in enumerate(tqdm(data)):
        temp=DGLMolTree(smiles)
        for key in temp.nodes_dict:
            result.update([temp.nodes_dict[key]['smiles']])
        # print('\r{}/{} smiles to get cliques vocabulary..'.format(i + 1, len(data)), end='')

    print('\n\tGet Vocabulary Finished!')

    with open(output_path,"w") as f:
        for csmiles in result:
            f.writelines(csmiles+"\n")


def get_slots(smiles):   #获取官能团块的原子特征
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):   #官能团词汇表hash函数，建立词表文件(txt):索引和内容的hash函数
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        # self.slots = [get_slots(smiles) for smiles in self.vocab]
        self.slots = None
        
    def get_index(self, smiles):
        if smiles not in self.vmap:
            print(f'{smiles} not in vocab.')

        return self.vmap.get(smiles,0)  # get the smiles. else unknow

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


class DGLMolTree(DGLGraph):
    def __init__(self, smiles):
        DGLGraph.__init__(self)
        self.nodes_dict = {}

        if smiles is None:
            return

        self.smiles = smiles
        self.mol = get_mol(smiles)

        # cliques: a list of list of atom indices
        # edges: a list of list of edge by atoms src and dst
        cliques, edges = tree_decomp(self.mol)

        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            csmiles = get_smiles(cmol)
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=get_mol(csmiles),
                clique=c,
            )
            if min(c) == 0:
                root = i

        self.add_nodes(len(cliques))

        # The clique with atom ID 0 becomes root
        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = self.nodes_dict[root][attr], self.nodes_dict[0][attr]

        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')
        for i, (_x, _y) in enumerate(edges):
            x = 0 if _x == root else root if _x == 0 else _x
            y = 0 if _y == root else root if _y == 0 else _y
            src[2 * i] = x
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x

        self.add_edges(src, dst)


    def treesize(self):
        return self.number_of_nodes()

