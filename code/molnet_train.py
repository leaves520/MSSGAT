import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from GNN_utils.ScaffoldSplit import scaffold_randomized_spliting_xiong
from GNN_utils.utils import show_figure_loss, get_valid, Model_molnet, set_seed

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dgl import batch
from GNN_utils.data_process_ours import multi_process, Dataset_multiprocess_ecfp, Collator_ecfp, \
    _compute_df_weights, sigleprocess_get_graph_and_save
from GNN_utils.pytorchtools import EarlyStopping
from GNN_utils.mol_tree import Vocab
from GNN_utils.nnutils_ours import GatEnconder_tree_gru, GatEncoder_raw_gru, \
    MLP_revised, Residual, FocalLoss, tree_gru,GIN_raw,raw_gru_adde,\
    tree_gru_onehot,raw_attention,raw_set2set,tree_gru_s2s,raw_gru_s2s,GatEncoder_raw_gru_s2s,tree_gru_onehot_s2s, \
    tri_gat,TrimNet, tree_gru_onehot_revised, GatEncoder_raw_gru_revised, MLP_residual


class All_old2(nn.Module):
    def __init__(self, hidden_state_size, vocab_path,nums_task, head, conv, rhead, rconv):
        super(All_old2, self).__init__()
        self.vocab = Vocab([x.strip("\r\n ") for x in open(vocab_path)])
        self.hidden_state_size = hidden_state_size


        self.GATencoder = tree_gru_onehot_revised(vocab=self.vocab, hidden_size=self.hidden_state_size,
                                          head_nums=head,conv_nums=conv)

        self.GATencoder_raw = GatEncoder_raw_gru_revised(hidden_size=self.hidden_state_size,
                                                 head_nums=rhead,conv_nums=rconv)

        self.dnn_ecfp = MLP_revised(n_feature=512, n_hidden=[256, 128],
                                    n_output=self.hidden_state_size, dropout=0.1)
        #
        self.classify = MLP_revised(self.hidden_state_size * 3, [32],  # [64,32],[128]
                                    2*nums_task, dropout=0.1)

   
        self.ln = nn.LayerNorm(self.hidden_state_size*3)
        self.relu = nn.ReLU()
        self.mol_output = None


    def forward(self, data, device):
        _, raw, fp = self.to_device(data, device)
        raw_h, x_r = self.GATencoder_raw(raw)
        tree = self.test_(data['mol_trees'],raw_h,device)
        self.mol_output, x_t = self.GATencoder(tree)
        x_fp = self.dnn_ecfp(fp)
        x = torch.cat([x_t, x_r, x_fp], dim=-1)
        y = self.classify(self.relu(self.ln(x)))
        return y



    def to_device(self, mol_batch, device):
        tree = batch(mol_batch['mol_trees']).to(device)
        raw = batch(mol_batch['mol_raws']).to(device)
        fp = mol_batch['ecfp'].to(device)

        return tree, raw, fp

    def test_(self,tree,raw_h,device):
        assert len(tree) == len(raw_h)
        all_data = []
        for i in range(len(raw_h)):
            tt = tree[i].nodes_dict
            r = raw_h[i]
            cliques = []
            for key in tt:
                clique = tt[key]['clique']
                cliques.append(torch.sum(r[clique],dim=0))
            try:
                all_data.append(torch.stack(cliques,dim=0))
            except:
                print(tree[i].smiles)
                all_data.append(torch.sum(r[:],dim=0))
                return

        assert len(all_data) == len(tree)
        for i in range(len(tree)):
            tree[i].ndata['h'] = all_data[i].cpu()

        return batch(tree).to(device)






def data_load():
    print("============Loading Data============")
    dataset_name = args['dataset_name']
    load_data = pd.read_csv(args['data_path'])

    if dataset_name == 'bace':
        tasks = load_data.columns.tolist()[2:3]
    elif dataset_name == 'bbbp':
        tasks = load_data.columns.tolist()[:1]
    elif dataset_name == 'clintox':
        tasks = load_data.columns.tolist()[1:]
    elif dataset_name == 'hiv':
        tasks = load_data.columns.tolist()[-1:]
    elif dataset_name == 'muv':
        tasks = load_data.columns.tolist()[:-2]
    elif dataset_name == 'sider':
        tasks = load_data.columns.tolist()[:-1]
    elif dataset_name == 'tox21':
        tasks = load_data.columns.tolist()[:-2]
    elif dataset_name == 'toxcast':
        tasks = load_data.columns.tolist()[1:]
    else:
        raise ValueError(f'Not support this {dataset_name}')

    try:
        load_data.rename(columns={'smiles': 'Smiles'}, inplace=True)
        print('rename the columns `smiles` to `Smiles`....')
    except:
        pass

    load_data = get_valid(load_data,shuffle=False,random_seed=args['seed'])
    y, task_weights = _compute_df_weights(load_data, tasks)
    print("The numbers of data= %s  \nThe dimension of label= %s" % (load_data.shape[0], y.shape))

    print("============Spliting Data============")
    if dataset_name in ['bace','bbbp','hiv']:
        print("Scaffold Split........")
        Train_index, Val_index, Test_index = scaffold_randomized_spliting_xiong(smiles_tasks_df=load_data, tasks=tasks,
                                                                          weights=task_weights,random_seed=args['seed'])

        print("============Processing Data============")
        multi_process(load_data.iloc[Train_index].Smiles, y[Train_index],
                      vocab_path=args['vocab'],data_type='train', data_name=dataset_name, reprocess=args['reprocess'])
        multi_process(load_data.iloc[Val_index].Smiles, y[Val_index],vocab_path=args['vocab'],
                      data_type='val', data_name=dataset_name, reprocess=args['reprocess'])
        multi_process(load_data.iloc[Test_index].Smiles, y[Test_index],vocab_path=args['vocab'],
                      data_type='test', data_name=dataset_name, reprocess=args['reprocess'])
    else:
        print("Random Split........")
        data = load_data['Smiles']
        X_data, X_test, y_data, y_test = train_test_split(data, y, test_size=0.1, random_state=args['seed'])
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=X_test.shape[0],
                                                          random_state=args['seed'])
        multi_process(X_train.tolist(), y_train.tolist(),
                      vocab_path=args['vocab'],data_type='train', data_name=dataset_name, reprocess=args['reprocess'])
        multi_process(X_val.tolist(), y_val.tolist(),
                      vocab_path=args['vocab'],data_type='val', data_name=dataset_name, reprocess=args['reprocess'])
        multi_process(X_test.tolist(), y_test.tolist(),
                      vocab_path=args['vocab'], data_type='test', data_name=dataset_name, reprocess=args['reprocess'])


    # Dataloader : generate enable iter object
    train_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='train')
    dataloader_trn = DataLoader(train_data, batch_size=args['bs'], shuffle=True, num_workers=0,
                                collate_fn=Collator_ecfp(), drop_last=False, )

    val_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='val')
    dataloader_val = DataLoader(val_data, batch_size=args['bs'], shuffle=False, num_workers=0,
                                collate_fn=Collator_ecfp(), drop_last=False, )

    test_data = Dataset_multiprocess_ecfp(data_name=dataset_name, data_type='test')
    dataloader_test = DataLoader(test_data, batch_size=args['bs'], shuffle=False, num_workers=0,
                                 collate_fn=Collator_ecfp(), drop_last=False, )

    return dataloader_trn, dataloader_val, dataloader_test, task_weights, tasks


def model_bulid(weights,tasks):
    # model defined
    print("============Building Model============")
    gcn = All_old2(hidden_state_size=args["hidden_state_size"], vocab_path=args['vocab'],nums_task=len(tasks),
                  head=args['head'],conv=args['conv'],rhead=args['rhead'],rconv=args['rconv'])

    print("Model #Params: %dK" % (sum([x.nelement() for x in gcn.parameters()]) / 1000,))

    device = args['gpu'] if torch.cuda.is_available() and args['gpu'] != -1 else 'cpu'
    optimizer = optim.Adam(gcn.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = [nn.CrossEntropyLoss(torch.tensor(weight, device=device), reduction='mean') for weight in weights]
    model = Model_molnet(model=gcn, optimizer=optimizer,
                  criterion=loss_function, scheduler=scheduler, device=device, tasks=tasks)

    print(f'use the device:{device}')

    return model


def train(model, trn, val, test):
    earlystop = EarlyStopping(args['patience'], verbose=True, dataset_name=dataset_name, model_save_path=args['save'])

    file = open(args['save'] + '/train_info.txt', 'w')
    best_roc = - float("inf")

    trn_list, val_list, test_list = [], [], []
    trn_roc_list, val_roc_list, test_roc_list = [], [], []
    for e in range(args['epochs']):
        trn_loss = model.train(trn)
        if args['trn']:
            _, trn_roc = model.eval(trn)
            trn_roc_list.append(trn_roc)

        val_loss, val_roc = model.eval(val)
        test_loss, test_roc = model.eval(test)

        date = f'epochs {e} --> loss: train {trn_loss}, val {val_loss}, ' \
               f'test {test_loss}\n val_roc:{val_roc} test_roc:{test_roc}\n'
        if e % 10 == 0:
            file.write(date)
        print(date, end='')

        trn_list.append(trn_loss)
        val_list.append(val_loss)
        test_list.append(test_loss)

        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)

        if args['earlystop']:
            earlystop(val_roc, model.model)
            if earlystop.early_stop:
                print('Early stopping')
                break
        else:
            if val_roc > best_roc:
                best_roc = val_roc
                print('save the best model....')
                torch.save(model.model.state_dict(), os.path.join(args['save'], 'best_model.pt'))

    best_index = np.argmax(val_roc_list)
    val_ = val_roc_list[best_index]
    test_ = test_roc_list[best_index]
    date = 'Finished! roc-score in test-data (best in val-data {}(epochs{})) is: {} -seed:{}'.format(val_, best_index, test_,args['seed'])
    print(date)

    file.write(date)
    file.flush()

    show_figure_loss(trn_list, val_list, test_list, trn_roc_list, val_roc_list, test_roc_list,
            save_path=args['save'],data_name=args['dataset_name'],trn_roc_verbose=args['trn'])

    all_res_path = os.path.split(args['save'])[0]
    with open(os.path.join(all_res_path, 'all_res.txt'), 'a+') as f:
        m = args['model']
        f.write(f'{test_} {m}'+ ' seed ' + str(args['seed']) + '\n')
        f.flush()


def main():
    trn, val, test, weights, tasks = data_load()
    model = model_bulid(weights,tasks)
    train(model, trn, val, test)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-path", required=True,
                        help='Path to a csv file for loading a dataset')
    parser.add_argument("-hs", "--hidden-state-size", type=int, default=64,
                        help="feature dimension of graph representation.(default:64)")
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpus used (default device:0, -1 is cpu)')
    parser.add_argument("--epochs", type=int, default=150,
                        help="Max number of epochs for training.(default:150)")
    parser.add_argument("--seed", default=1, type=int,
                        help='random-seed for reproduct')
    parser.add_argument('--reprocess', default=False, action='store_true',
                        help='whether reprocess the data or not.(default:False)')
    parser.add_argument('--earlystop', default=False, action='store_true',
                        help='whether use earlystop or not.(default:False)')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience when training if `earlystop` is used')
    parser.add_argument('--workers', type=int, default=8,
                        help='workers for data pre-process')
    parser.add_argument('--trn', default=False, action='store_true',
                        help='whether verbose the training roc')
    parser.add_argument('--save', default='./result_save', help='path to save model and result')
    parser.add_argument('--head', default=4, type=int)
    parser.add_argument('--conv', default=2, type=int)
    parser.add_argument('--rhead', default=4, type=int)
    parser.add_argument('--rconv', default=3, type=int)
    parser.add_argument('--bs', default=256, type=int)

    args = parser.parse_args().__dict__

    if not os.path.exists(args["data_path"]):
        raise ValueError("Path to dataset file not found , plz check the path again ")

    dataset_name = os.path.split(args["data_path"])[-1].split('.')[0]
    args['dataset_name'] = dataset_name

    args['save'] = os.path.join(args['save'], dataset_name)
    args['save'] = os.path.join(args['save'], 'process-{}-{}'.format('all', time.strftime('%y%m%d%H%M')))
    if not os.path.exists(args['save']):
        os.makedirs(args['save'])

    vocab_path = './dataset/vocabulary_molnet.txt'
    args['vocab'] = vocab_path
    set_seed(args['seed'])

    import json
    with open(args['save'] + '/config.json', 'w') as f:
        json.dump(args, f)

    main()
