import os,pickle,multiprocessing,torch
import shutil

import dgl
import numpy as np
from dgl import DGLGraph
from torch.utils.data import Dataset
from tqdm import tqdm

from .mol_tree import Vocab, DGLMolTree
from .chemutils import atom_features, get_mol, get_morgan_fp, bond_features,get_dgl_node_feature,get_dgl_bond_feature


def _unpack_field(examples, field):  # get batch examples (dictionary) values by key
    return [e[field] for e in examples]

def _set_node_id(mol_tree, vocab):  #hash函数，找到mol_tree中每个顶点(官能团簇)在vocab中的索引
    wid = []
    for i, node in enumerate(mol_tree.nodes_dict):
        mol_tree.nodes_dict[node]['idx'] = i
        wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))
    return wid



# single_process for data preprocess
def sigleprocess_get_graph_and_save(X, y, vocab_path, data_type, data_name=None,reprocess = False ):
    assert data_name is not None
    assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

    save_path = './dataset/graph_data_ours'
    data_path = save_path + '/' + data_name + '_' + data_type + '.p'

    if os.path.exists(data_path) and not reprocess:
        print(data_name + '_' + data_type + ' is already finshed')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vocab = Vocab([x.strip("\r\n ") for x in open(vocab_path)])
    X_zip = list(zip(X, y))
    return_list = []
    for idx, (smiles, label) in enumerate(X_zip):
        # mol_tree = DGLMolTree(smiles)  # mol_tree
        # wid = _set_node_id(mol_tree, vocab)  # idx_cliques

        # get the raw mol graph
        mol = get_mol(smiles)

        feats = get_dgl_node_feature(mol)
        mol_raw = DGLGraph()
        mol_raw.add_nodes((len(mol.GetAtoms())))
        mol_raw.ndata['h'] = feats

        for bonds in mol.GetBonds():
            src_id = bonds.GetBeginAtomIdx()
            dst_id = bonds.GetEndAtomIdx()
            mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])

        mol_raw = dgl.add_self_loop(mol_raw)
        e_f = get_dgl_bond_feature(mol)
        mol_raw.edata['e_f'] = e_f
        result = {'mol_raw': mol_raw,'label':label}
        return_list.append(result)
        print('\r{}/{} molecules to process..'.format(idx + 1, len(X)), end='')

    print("\n  Process of {} is Finshed".format(data_name + '_' + data_type))

    with open(data_path, 'wb') as file:
        pickle.dump(return_list, file)
    file.close()



# multi_process for data preprocess add ecfp
def get_batchs(X,vocab,data_path):
    return_res = []
    for idx,(smiles,label) in enumerate(X):
        mol_tree = DGLMolTree(smiles)  # mol_tree
        mol_tree = dgl.add_self_loop(mol_tree)  # mol_tree 是否加自环边
        wid = _set_node_id(mol_tree, vocab)  # idx_cliques

        # get the raw mol graph
        atom_list = []
        mol = get_mol(smiles)

        # # original 44 feat size
        # for atoms in mol.GetAtoms():
        #     atom_f = atom_features(atoms)  # one-hot features for atoms
        #     atom_list.append(atom_f)
        # atoms_f = np.vstack(atom_list)  # get the atoms features
        #
        # mol_raw = DGLGraph()
        # mol_raw.add_nodes(len(mol.GetAtoms()))
        # mol_raw.ndata['h'] = torch.Tensor(atoms_f)


        ## dgl 74 feat size  revise by 6/26
        feats = get_dgl_node_feature(mol)
        mol_raw = DGLGraph()
        mol_raw.add_nodes((len(mol.GetAtoms())))
        mol_raw.ndata['h'] = feats


        for bonds in mol.GetBonds():
            src_id = bonds.GetBeginAtomIdx()
            dst_id = bonds.GetEndAtomIdx()
            mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])

        mol_raw = dgl.add_self_loop(mol_raw)
        e_f = get_dgl_bond_feature(mol)
        mol_raw.edata['e_f'] = e_f


        # ############################ add e_features
        # edges_list = []
        # for bonds in mol.GetBonds():
        #     src_id = bonds.GetBeginAtomIdx()
        #     dst_id = bonds.GetEndAtomIdx()
        #     mol_raw.add_edges([src_id, dst_id], [dst_id, src_id])
        #     edges_list.append(bond_features(bonds, self_loop=True))
        #     edges_list.append(bond_features(bonds, self_loop=True))
        #
        # for idx in range(len(mol.GetAtoms())):
        #     mol_raw.add_edges([idx],[idx])  # add self-loop
        #     edges_list.append(np.array([.0]*10 + [1.]))
        #
        # edges_f = np.vstack(edges_list)
        # mol_raw.edata['e_f'] = torch.Tensor(edges_f)
        # #####################  add edge features

        ecfp = get_morgan_fp(smiles)
        result = {'mol_tree': mol_tree, 'wid': wid, 'mol_raw': mol_raw,'label':label,'fp':ecfp}
        return_res.append(result)

        print('\r{}/{} molecules to process..'.format(idx + 1, len(X)), end='')

    id = os.getpid()
    with open(data_path + f'/{id}.pkl', 'wb') as file:
        pickle.dump(return_res, file)


def multi_process(X,y,data_type,vocab_path, data_name = None,workers=8,reprocess = False):
    assert data_name is not None
    assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

    if workers == -1:
        workers = multiprocessing.cpu_count()

    print(f"Use {workers} cpus to process: ")

    save_path = './dataset/graph_data_ours'
    data_path = save_path + '/' + data_name + '_' + data_type + '.p'

    if os.path.exists(data_path) and not reprocess:
        print(data_name + '_' + data_type + ' is already finshed')
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tmp_path = './tmp'
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(tmp_path)

    vocab = Vocab([x.strip("\r\n") for x in open(vocab_path)])
    X_zip = list(zip(X, y))
    radio = len(X_zip)//workers # split the data_set for multiprocess
    jobs = []
    for i in range(workers):
        if i == workers-1:
            X = X_zip[i * radio:]
        else:
            X = X_zip[i * radio:(i+1) * radio]
        s = multiprocessing.Process(target=get_batchs,kwargs={'X':X,'vocab':vocab,'data_path':tmp_path})
        jobs.append(s)
        s.start()

    for proc in jobs:
        proc.join()    # join process until the main process finished

    print("\n  Process of {} is Finshed".format(data_name + '_' + data_type))

    concat_result = []
    for name in tqdm(os.listdir(tmp_path),'===process'):
        with open(tmp_path + f'/{name}','rb') as file:
            d = pickle.load(file)
        concat_result.extend(d)

    with open(data_path,'wb') as file:
        pickle.dump(concat_result,file)

    shutil.rmtree(tmp_path)


class Dataset_multiprocess_ecfp(Dataset):   # torch.dataset is abstract class, need to override the __len__ and __getitem__
    def __init__(self,data_name, data_type,load_path='./dataset/graph_data_ours'):
        self.data_name = data_name
        self.data_path = load_path + '/' + data_name + '_' + data_type + '.p'

        assert os.path.exists(self.data_path),"not exists the path to data.p"
        assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

        self.data_and_label = pickle.load(open(self.data_path,'rb'))

    def __len__(self):
        return len(self.data_and_label)

    def __getitem__(self, idx):
        '''get datapoint with index'''
        mol_tree = self.data_and_label[idx]['mol_tree']
        wid = self.data_and_label[idx]['wid']
        mol_raw = self.data_and_label[idx]['mol_raw']
        label = self.data_and_label[idx]['label']
        fps = self.data_and_label[idx]['fp']

        result = {'mol_tree': mol_tree, 'wid': wid, 'class': label, 'mol_raw': mol_raw,'fps':fps}

        return result


class Collator_ecfp(object):   # get the batch_data as input, __call__ must be implement
    '''get list of trees and label'''
    def __call__(self, examples):
        mol_trees = _unpack_field(examples, 'mol_tree')
        wid = _unpack_field(examples, 'wid')
        label = _unpack_field(examples, 'class')
        mol_raws=_unpack_field(examples,'mol_raw')
        ecfp = _unpack_field(examples,'fps')

        # wid is the index of cliques on word embedding matrix
        for _wid, mol_tree in zip(wid, mol_trees):  # zip wid,mol_trees,label as tuple
             mol_tree.ndata['wid'] = torch.LongTensor(_wid)

        ecfp = torch.Tensor(np.vstack(ecfp))

        batch_data = {'mol_trees': mol_trees,'class':np.array(label),
                      'mol_raws': mol_raws,'ecfp':ecfp}

        return batch_data


class Dataset_others(Dataset):   # torch.dataset is abstract class, need to override the __len__ and __getitem__
    def __init__(self,data_name, data_type,load_path='./dataset/graph_data_ours'):
        self.data_name = data_name
        self.data_path = load_path + '/' + data_name + '_' + data_type + '.p'

        assert os.path.exists(self.data_path),"not exists the path to data.p"
        assert data_type in ['train', 'val', 'test'], "data_type must in choices ['train','val','test']"

        self.data_and_label = pickle.load(open(self.data_path,'rb'))

    def __len__(self):
        return len(self.data_and_label)

    def __getitem__(self, idx):
        '''get datapoint with index'''
        mol_raw = self.data_and_label[idx]['mol_raw']
        label = self.data_and_label[idx]['label']

        result = {'class': label, 'mol_raw': mol_raw}

        return result


class Collator_others(object):   # get the batch_data as input, __call__ must be implement
    '''get list of trees and label'''
    def __call__(self, examples):
        label = _unpack_field(examples, 'class')
        mol_raws=_unpack_field(examples,'mol_raw')

        batch_data = {'class':np.array(label),
                      'mol_raws': mol_raws}

        return batch_data


# compute the weights for each label , in order to avoid unbalanced datas
def _compute_df_weights(df, tasks):
    weights = []
    for i, task in enumerate(tasks):
        negative_df = df[df[task] == 0][["Smiles", task]]
        positive_df = df[df[task] == 1][["Smiles", task]]
        try:
            weights.append([(positive_df.shape[0] + negative_df.shape[0]) / negative_df.shape[0], \
                            (positive_df.shape[0] + negative_df.shape[0]) / positive_df.shape[0]])  # 计算正负样本比例权重
        except:
            weights.append([1.0,1.0])

    n_samples = df.shape[0]
    y = np.hstack([np.reshape(np.array(df[task].values), (n_samples, 1)) for task in tasks])

    return y.astype(float), weights

def regression_y(df, tasks):
    y = df[tasks].values
    return y.astype(float)
