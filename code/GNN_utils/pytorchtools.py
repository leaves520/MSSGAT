import numpy as np
import torch
import os

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, dataset_name=None,model_save_path=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.roc_max = - np.Inf
        self.delta = delta

        self.dataset_name = dataset_name
        self.model_sava_path = model_save_path
        self.best_model_path = None

    def __call__(self, roc_score, model):

        score = roc_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(roc_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(roc_score, model)
            self.counter = 0

    def save_checkpoint(self, roc_score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation roc-score increased ({self.roc_max:.6f} --> {roc_score:.6f}).  Saving model ...')
        self.best_model_path = os.path.join(self.model_sava_path, 'Best_model_' + self.dataset_name + '.pt')
        torch.save(model.state_dict(), self.best_model_path)
        self.roc_max = roc_score
