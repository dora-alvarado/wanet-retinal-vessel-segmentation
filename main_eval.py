from trainer.dataset import RetinaDataset
from evaluation.evaluation import Evaluation
from model.wanet import WANet
from utils.misc_img_func import paste_imgs
from utils.settings import setup_parser, save_config
from trainer.dataset_config_file import get_dataset_settings

import numpy as np
import os
import torch
import cv2


if __name__=='__main__':
    config = setup_parser()
    save_config(config, config.experiment + 'test_config.txt')
    dataset_config = get_dataset_settings(config.dataset)(config.path_dataset)

    listOfFiles_imgs = np.sort(np.asarray(os.listdir(dataset_config['path_test_imgs'])))
    listOfFiles_imgs = np.asarray([name for name in listOfFiles_imgs if name.endswith(dataset_config['image_ext'])])
    print(listOfFiles_imgs)

    dataset = RetinaDataset(dataset_config['path_test_imgs'],
                            dataset_config['path_test_gts'],
                            listOfFiles_imgs,
                            img_ext=dataset_config['image_ext'],
                            gt_ext=dataset_config['gt_ext'],
                            transform=False,
                            fov_dpath=dataset_config['path_test_fovs'],
                            fov_ext=dataset_config['fov_ext'])

    # model settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # initialize model
    model = WANet(n_channels=1, n_classes=2)

    # load state
    experiment_path = config.experiment
    model_path = experiment_path + 'model_best.pth.tar' if config.best else experiment_path + 'model_checkpoint.pth.tar'
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    # evaluate model
    eval = Evaluation(model, config.crop_size, stride=config.stride, logfile_path=experiment_path+'performance.log')
    eval(dataset, config.num_imgs)
    n_rows = round(config.num_imgs / config.num_group)
    print(n_rows, config.num_group)
    all_originals = paste_imgs((dataset.lst_imgs[:config.num_imgs]*255).astype(np.uint8), n_rows=n_rows, n_cols=config.num_group, sep=0)
    all_gts = paste_imgs((dataset.lst_gts[:config.num_imgs]*255).astype(np.uint8), n_rows=n_rows, n_cols=config.num_group, sep=0)
    all_fovs = paste_imgs((dataset.lst_fovs[:config.num_imgs]*255).astype(np.uint8), n_rows=n_rows, n_cols=config.num_group, sep=0)
    all_predictions = paste_imgs((eval.lst_predictions*255).astype(np.uint8), n_rows=n_rows, n_cols=config.num_group, sep=0)

    cv2.imwrite(experiment_path + 'all_originals.png', all_originals)
    cv2.imwrite(experiment_path + 'all_fovs.png', all_fovs)
    cv2.imwrite(experiment_path + 'all_groundtruths.png', all_gts)
    cv2.imwrite(experiment_path + 'all_predictions.png', all_predictions)