from collections import defaultdict
from ogb.graphproppred import Evaluator
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，为所有的GPU设置种子。
    torch.backends.cudnn.deterministic = True  # CPU和GPU结果一致
    torch.backends.cudnn.benchmark = False



def show_figure_loss(train_loss, val_loss, test_loss, trn_roc, val_roc, test_roc, save_path, data_name,
                     trn_roc_verbose=False):

    save_path = os.path.join(save_path,data_name)

    plt.figure('Training Process')  # Create art board
    plt.plot(train_loss, 'r-', label='train loss', )
    plt.plot(val_loss, 'b-', label='validation loss', )
    #plt.plot(test_loss, 'g-', label='test loss')

    plt.title("Training Process For " + data_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.savefig(save_path + '_loss.pdf')

    plt.cla()

    plt.figure('Training Process')  # Create art board
    if trn_roc_verbose:
        plt.plot(trn_roc, 'r-', label='train roc-auc', )
    plt.plot(val_roc, 'b-', label='validation roc-auc', )
    #plt.plot(test_roc, 'g-', label='test roc-auc')

    plt.title("Training Process For " + data_name)
    plt.xlabel('Epochs')
    plt.ylabel('ROC-AUC')
    plt.legend(loc=0)
    plt.savefig(save_path + '_roc.pdf')



def get_valid(load_data, shuffle=False, random_seed=42):
    from rdkit import Chem
    # get the valid mol
    not_sucessful = 0
    smilesList = load_data.Smiles.values
    print("number of all smiles: ", len(smilesList))
    remained_smiles = []
    canonical_smiles_list = []
    for smiles in tqdm(smilesList, desc='===Check the valid data'):
        try:
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)
        except:
            not_sucessful += 1
            pass
    print("number of failed :", not_sucessful)
    print("number of successfully processed smiles: ", len(remained_smiles))
    load_data = load_data[load_data["Smiles"].isin(remained_smiles)].reset_index()

    if shuffle:
        load_data = load_data.sample(frac=1, random_state=random_seed)  # shuffle the data

    return load_data



class Model_molnet(object):
    def __init__(self, model, optimizer, criterion, scheduler, device, tasks):
        assert isinstance(model, nn.Module)
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.tasks = tasks

    def train(self, dataloader):
        # Training
        loss_list = []
        self.model.train()
        for it, batchs in enumerate(tqdm(dataloader, desc='===Training process')):
            logits = nn.Softmax()(self.model(batchs,self.device))
            loss = 0.0
            for i, task in enumerate(self.tasks):
                y_pred = logits[:, i * 2:(i + 1) * 2]
                y_val = batchs['class'][:, i]
                vaildInds = (np.where((y_val == 0) | (y_val == 1))[0]).astype('int64')
                if len(vaildInds) == 0:
                    continue
                y_val_adjust = np.array([y_val[v] for v in vaildInds]).astype('int64')
                vaildInds = torch.tensor(vaildInds,device=self.device).squeeze()
                y_pred_adjust = torch.index_select(y_pred, 0, vaildInds)
                loss += self.criterion[i](y_pred_adjust, torch.tensor(y_val_adjust,device=self.device))

            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # update

            loss_list.append(loss.cpu().detach().numpy())

        self.scheduler.step()

        return np.array(loss_list).mean()

    def eval(self, dataloader):
        y_true_list = defaultdict(list)
        y_pred_list = defaultdict(list)
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for it, batchs in enumerate(dataloader):
                logits = nn.Softmax()(self.model(batchs, self.device))
                loss = 0.0
                for i, task in enumerate(self.tasks):
                    y_pred = logits[:, i * 2:(i + 1) * 2]
                    y_val = batchs['class'][:, i]
                    vaildInds = (np.where((y_val == 0) | (y_val == 1))[0]).astype('int64')
                    if len(vaildInds) == 0:
                        continue
                    y_val_adjust = np.array([y_val[v] for v in vaildInds]).astype('int64')
                    vaildInds = torch.tensor(vaildInds, device=self.device).squeeze()
                    y_pred_adjust = torch.index_select(y_pred, 0, vaildInds)
                    loss += self.criterion[i](y_pred_adjust, torch.tensor(y_val_adjust, device=self.device))

                    pred_positive = y_pred_adjust[:, 1].data.cpu().numpy()
                    y_true_list[i].extend(y_val_adjust)
                    y_pred_list[i].extend(pred_positive)

                loss_list.append(loss.cpu().detach().numpy())


            all_eval_roc = []
            for i in range(len(self.tasks)):
                assert len(y_true_list[i]) == len(y_pred_list[i])
                evaluator = Evaluator(name='ogbg-molhiv')
                input_dict = {'y_true': np.array(y_true_list[i]).reshape(-1, 1),
                              'y_pred': np.array(y_pred_list[i]).reshape(-1, 1)}
                try:
                    result_dict = evaluator.eval(input_dict)
                    eval_roc = list(result_dict.values())[0]
                    all_eval_roc.append(eval_roc)
                except:
                    pass

        return np.array(loss_list).mean(), np.array(all_eval_roc).mean()


class Model_rmse(object):
    def __init__(self,model, optimizer, criterion, scheduler,device):
        assert isinstance(model,nn.Module)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    def train(self,dataloader):
        # Training
        loss_list = []
        self.model.train()
        for it, batchs in enumerate(tqdm(dataloader, desc='===Training process')):
            logits = self.model(batchs, self.device)
            y_pred = logits
            y_val = np.array(batchs['class'],dtype=np.float32)
            loss = self.criterion(y_pred, torch.tensor(y_val,device=self.device))

            self.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # update

            loss_list.append(loss.cpu().detach().numpy())

        self.scheduler.step()

        return np.array(loss_list).mean()

    def eval(self,dataloader):
        from math import sqrt
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for it, batchs in enumerate(dataloader):
                logits = self.model(batchs, self.device)
                y_pred = logits
                y_val = np.array(batchs['class'],dtype=np.float32)
                loss = self.criterion(y_pred, torch.tensor(y_val,device=self.device))
                loss_list.append(sqrt(loss.cpu().detach().numpy()))

        return np.mean(np.array(loss_list)**2), np.array(loss_list).mean()
