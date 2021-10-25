from trainer.dataset_config_file import drive_settings
from trainer.dataset import RetinaDataset
from evaluation.evaluation import Evaluation
from model.wanet import WANet
from utils.misc_img_func import paste_imgs
import numpy as np
import os
import torch
import cv2

# dataset settings
drive_path = 'datasets/'
dataset_config = drive_settings(drive_path)

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
patch_size = 48
batch_size = 32
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# initialize model
model = WANet(n_channels=1, n_classes=2)

# load state
experiment_path = './exp2/'
model_path = experiment_path + 'model_best.pth.tar'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])

N_imgs = 20

eval = Evaluation(model, patch_size, batch_size=batch_size, stride=5)
eval(dataset, N_imgs)

all_originals = paste_imgs((dataset.lst_imgs*255).astype(np.uint8), 4, 5, sep=0)
all_gts = paste_imgs((dataset.lst_gts*255).astype(np.uint8), 4, 5, sep=0)
all_fovs = paste_imgs((dataset.lst_fovs*255).astype(np.uint8), 4, 5, sep=0)
all_predictions = paste_imgs((eval.lst_predictions*255).astype(np.uint8), 4, 5, sep=0)

cv2.imwrite(experiment_path + 'all_originals.png', all_originals)
cv2.imwrite(experiment_path + 'all_fovs.png', all_fovs)
cv2.imwrite(experiment_path + 'all_groundtruths.png', all_gts)
cv2.imwrite(experiment_path + 'all_predictions.png', all_predictions)