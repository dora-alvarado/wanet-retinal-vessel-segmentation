import torch
import numpy as np
import random
import os
import shutil
from .metrics import accuracy


class Trainer(object):

    def __init__(self, model, optimizer, criteria=torch.nn.CrossEntropyLoss, metric=accuracy, scheduler=None, seed=0):
        self.model = model
        self.optimizer = optimizer
        self.criteria = criteria
        self.metric = metric
        self.scheduler = scheduler
        self.seed = seed
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.best_loss = float('Inf')
        self.best_metric = 0
        self.start_epoch = 0
        self.manual_seed()

    def manual_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        np.random.seed(self.seed)  # Numpy module.
        random.seed(self.seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_device(self):
        self.model = self.model.to(self.device)
        self.criteria = self.criteria.to(self.device)

    def train_step(self, dataloader):
        epoch_loss = 0.
        epoch_metric = 0.
        self.model.train()
        for X, y in dataloader:
            X = X.float().to(self.device)
            y = y.long().to(self.device)
            self.optimizer.zero_grad()

            y_pred_thin, y_pred_thick, y_pred = self.model(X)

            loss = self.criteria(y_pred, y) + 0.5 * (self.criteria(y_pred_thin, y) + self.criteria(y_pred_thick, y))
            metric = self.metric(y_pred, y)
            # update
            loss.backward()
            self.optimizer.step()
            if self.scheduler and hasattr(self.scheduler.__class__, 'batch_step') \
                    and callable(getattr(self.scheduler.__class__, 'batch_step')):
                self.scheduler.batch_step()

            epoch_loss += loss.item()
            epoch_metric += metric
        if self.scheduler:
            self.scheduler.step()

        epoch_loss = epoch_loss / len(dataloader)
        epoch_metric = epoch_metric / len(dataloader)
        return epoch_loss, epoch_metric

    def val_step(self, dataloader):
        epoch_loss = 0.
        epoch_metric = 0
        self.model.eval()
        for X, y in dataloader:
            X = X.float().to(self.device)
            y = y.long().to(self.device)

            with torch.no_grad():
                y_pred_thin, y_pred_thick, y_pred = self.model(X)
                loss = self.criteria(y_pred, y)+ 0.5*(self.criteria(y_pred_thin, y) + self.criteria(y_pred_thick, y))
                metric = self.metric(y_pred, y)
                epoch_loss += loss.item()
                epoch_metric += metric
        epoch_loss = epoch_loss / len(dataloader)
        epoch_metric = epoch_metric/ len(dataloader)
        return epoch_loss, epoch_metric

    def save_checkpoint(self, epoch, model_dir, is_best=False):
        # create the state dictionary
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_metric': self.best_metric,
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
        }
        # save
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = model_dir + "/model_checkpoint.pth.tar"

        torch.save(state, model_path)
        if is_best:
            shutil.copyfile(model_path, model_dir + '/model_best.pth.tar')

    def resume(self, model_dir):
        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        self.set_device()
        self.model_dir = model_dir
        if os.path.isfile(model_dir):
            print("=> loading checkpoint '{}'".format(model_dir))
            state = torch.load(model_dir, map_location=torch.device('cpu'))
            self.start_epoch = state['epoch']+1
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_metric = state['best_metric']
            self.best_loss = state['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_dir, state['epoch']+1))
        else:
            print("=> no checkpoint found at '{}'".format(model_dir))

    def __call__(self, dataloaders, n_epochs, model_dir):
        self.set_device()
        self.model_dir = model_dir
        for epoch in range(self.start_epoch, n_epochs):
            train_loss, train_metric = self.train_step(dataloaders['train'])
            val_loss, val_metric = self.val_step(dataloaders['val'])
            self.save_checkpoint(epoch, model_dir)
            if val_loss < self.best_loss:
                print('val_loss improved from %.4f to %.4f, saving model to path' % (self.best_loss, val_loss))
                self.best_loss = val_loss
                self.best_metric = val_metric
                self.save_checkpoint(epoch, model_dir, is_best=True)
            epoch_data = {'loss': train_loss,
                          'metric': train_metric,
                          'val_loss': val_loss,
                          'val_metric': val_metric,
                          'best_val_loss': self.best_loss,
                          'best_val_metric': self.best_metric}

            str1 = [' %s: %.4f ' % item for item in epoch_data.items()]
            print('Epoch %04d/%04d: %s' % (epoch + 1, n_epochs, '-'.join(str1)))

        if self.use_cuda:
            torch.cuda.empty_cache()



