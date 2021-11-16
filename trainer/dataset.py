import os
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from .preprocessing import read_grayscale_img, clahe_equalized, adjust_gamma, change_range
from torch import squeeze


class RetinaDataset(Dataset):

    def __init__(self,
                 img_dpath,
                 gt_dpath,
                 img_fnames,
                 n_patches=20,
                 img_ext = '.tif',
                 gt_ext ='_manual1.gif',
                 patch_size = 48,
                 transform = False,
                 fov_dpath = None,
                 fov_ext='.png',
                 ):

        self.img_dpath = img_dpath
        self.gt_dpath = gt_dpath
        self.fov_dpath = fov_dpath

        self.img_fnames = img_fnames
        self.gt_ext = gt_ext
        self.fov_ext = fov_ext
        self.patch_size = patch_size
        self.n_patches = n_patches

        self.lst_imgs = []
        self.lst_gts = []
        self.lst_fovs = []
        self.n_channels = None
        self.flag_transform = transform

        self.lst_img_filenames = [name for name in img_fnames if name.endswith(img_ext)]

        for fname_img in self.lst_img_filenames:
            # get name and extension
            fname, fname_ext = os.path.splitext(fname_img)
            # get full path to original image
            img_fpath = os.path.join(self.img_dpath, fname_img)
            # get full path to groud-truth mask
            fname_gt = fname+self.gt_ext
            gt_fpath = os.path.join(self.gt_dpath, fname_gt)
            # read ground-truth mask
            mask = read_grayscale_img(gt_fpath)/255.
            # read image
            img = read_grayscale_img(img_fpath)
            img = self.data_normalization(img, max_val =255, dtype=np.uint8)
            # preprocessing step
            img[:, :, 0] = self.preproc(img[:, :, 0])

            if self.fov_dpath is not None:
                fname_fov  = fname + self.fov_ext
                fov_fpath = os.path.join(self.fov_dpath, fname_fov)
                fov = read_grayscale_img(fov_fpath)/255.
                self.lst_fovs.append(fov)

            # save in RAM
            self.lst_imgs.append(img)
            self.lst_gts.append(mask)

        self.lst_imgs = np.asarray(self.lst_imgs, dtype=np.float64)
        self.lst_gts = np.asarray(self.lst_gts, dtype=np.float64)
        self.lst_fovs = np.asarray(self.lst_fovs, dtype=np.float64)
        # global normalization
        self.lst_imgs = self.data_normalization(self.lst_imgs, max_val=1., dtype=np.float64)
        self.n_imgs = len(self.lst_img_filenames)

    def preproc(self, image):
        image = clahe_equalized(image)
        image = adjust_gamma(image, 1.2)
        return image

    def data_normalization(self, img, max_val=255., dtype=np.uint8):
        img_std = np.std(img)
        img_mean = np.mean(img)
        img_normalized = (img - img_mean) / img_std
        img_normalized = change_range(img_normalized, img_normalized.min(), img_normalized.max(), 0., max_val)

        return img_normalized.astype(dtype)

    def transform(self, image, mask, prob=0.):
        # Border crop
        h, w, _ = image.shape
        min_dim2 = (np.max([h, w]) - np.min([h, w])) // 2
        if w < h:
            image = image[min_dim2:-min_dim2]  # cut bottom and top
            mask = mask[min_dim2:-min_dim2]  # cut bottom and top
        else:
            image = image[:, min_dim2:-min_dim2]  # cut left and right
            mask = mask[:, min_dim2:-min_dim2]  # cut left and right

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        # Random patch crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.patch_size, self.patch_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        # Random horizontal flipping
        if random.random() > (1.-prob):
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > (1.-prob):
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > (1. - prob):
            angle = transforms.RandomRotation.get_params(degrees=(0, 180))
            image = TF.rotate(image, angle=angle)
            mask = TF.rotate(mask, angle=angle)

        # Random Gaussian blur
        if random.random() > (1. - prob):
            sigma = transforms.GaussianBlur.get_params(0.1, 5)
            # only applied to image, mask should remain the same
            image = TF.gaussian_blur(image, kernel_size=(5,5), sigma=(sigma, sigma))

        return image, mask

    def __getitem__(self, i):

        i = i % self.n_imgs  # equally extracting patches from each image
        img = self.lst_imgs[i]
        mask = self.lst_gts[i]
        img_h, img_w, _ = img.shape
        self.img_h = img_h
        self.img_w = img_w

        prob = 0.3
        patch, patch_mask = self.transform(img, mask, prob=self.flag_transform * prob)
        # remove dim for patch_mask
        patch_mask = squeeze(patch_mask, dim=0)
        return patch, patch_mask

    def __len__(self):
        return self.n_patches

