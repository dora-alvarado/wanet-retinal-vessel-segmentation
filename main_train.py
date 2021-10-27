import os
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from trainer.dataset import RetinaDataset
from trainer.dataset_config_file import get_dataset_settings
from trainer.optimizer import optimizer_func as optim_selector
from trainer.trainer import Trainer
from model.wanet import WANet
from trainer.adamwr.cyclic_scheduler import CyclicLRWithRestarts
from utils.settings import setup_parser


if __name__=='__main__':
    config = setup_parser()
    dataset_config = get_dataset_settings(config.dataset)(config.path_dataset)

    listOfFiles_imgs = np.sort(np.asarray(os.listdir(dataset_config['path_train_imgs'])))
    listOfFiles_imgs = np.asarray([name for name in listOfFiles_imgs if name.endswith(dataset_config['image_ext'])])

    percent_validation = 0.1
    all_idx = np.asarray(range(len(listOfFiles_imgs)), dtype=np.int)
    idx_train_imgs, idx_val_imgs = train_test_split(all_idx, test_size=percent_validation, random_state=0)

    img_filenames = {
        'train': listOfFiles_imgs[idx_train_imgs],
        'val': listOfFiles_imgs[idx_val_imgs]
    }

    n_patches = config.num_patches
    patches = {'train': int(round(n_patches * (1. - percent_validation))),
               'val': int(round(n_patches * percent_validation))
               }

    datasets = {
        x: RetinaDataset(dataset_config['path_train_imgs'],
                         dataset_config['path_train_gts'],
                         img_filenames[x],
                         patch_size=config.crop_size,
                         img_ext=dataset_config['image_ext'],
                         gt_ext=dataset_config['gt_ext'],
                         n_patches=patches[x],
                         transform=False)
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(datasets[x], batch_size=config.batch_size, shuffle=True) for x in ['train', 'val']
        for x in ['train', 'val']
    }

    # initialize model
    model = WANet(n_channels=1, n_classes=2)

    # set loss function and optimizer
    criteria = CrossEntropyLoss()
    optimizer = optim_selector(config.optimizer, model, learning_rate=config.learning_rate)
    scheduler = CyclicLRWithRestarts(optimizer, config.batch_size, patches['train'], restart_period=10, t_mult=1.2,
                                     policy="cosine")
    # other schedulers can be used
    # from torch.optim.lr_scheduler import CyclicLR
    # scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, cycle_momentum=False)

    # train the model
    trainer = Trainer(model, optimizer, criteria=criteria, scheduler=scheduler, seed=0)
    if config.resume:
        checkpoint_path = config.experiment + 'model_checkpoint.pth.tar'
        trainer.resume(checkpoint_path)

    trainer(dataloaders, config.num_epochs, config.experiment)
