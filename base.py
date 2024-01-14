import abc
from dataloader.data_utils import *
from utils import (
    Averager, Timer
)

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()
        self.init_log()

    @abc.abstractmethod
    def train(self):
        pass
    
    def init_log(self):
        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * self.args.sessions

        self.trlog['seen_acc'] = []
        self.trlog['unseen_acc'] = []