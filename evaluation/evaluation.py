import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
from .extract_patches import patches_overlap, recompone_overlap
from .performance import performance
import logging

class Evaluation(object):

    def __init__(self, model, patch_size, batch_size=32, stride=5):
        self.model = model
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.stride = stride
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.lst_predictions = None

    def init_logger(self, logfile_path):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.FileHandler(logfile_path, mode='w', encoding='utf-8')
        ch.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter('%(message)s')
        # add formatter to ch
        ch.setFormatter(formatter)
        # add ch to logger
        self.logger.addHandler(ch)

    def chunk(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def predict_img(self, img):
        ch, h, w = img.shape
        model = self.model.to(self.device)
        model = model.eval()
        patches, new_h, new_w = patches_overlap(img, self.patch_size, self.stride)
        y_pred = []
        with torch.no_grad():
            for X in self.chunk(patches, self.batch_size):
                out1, out2, out3 = model(X.float().to(self.device))
                out = F.softmax(out3, dim=1).cpu().numpy()  # outputs
                y_pred.append(out)
        torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred, axis=0)
        y_pred = y_pred[:, 1:2, : , :] # class 1 -vessels
        pred_img = recompone_overlap(y_pred, new_h, new_w, self.stride)  # predictions
        return pred_img[0, 0, :h, :w]

    def eval_performance(self, dataset, inside_FoV=False):
        y_scores = []
        y_true =[]
        n_imgs = len(self.lst_predictions)

        for i in range(n_imgs):
            pred = self.lst_predictions[i].reshape(-1)

            true = dataset.lst_gts[i].reshape(-1)
            if inside_FoV:
                fov = dataset.lst_fovs[i].reshape(-1)
                pred = pred[fov>0]
                true = true[fov>0]

            y_scores+=list(pred)
            y_true+=list(true)

        y_scores = np.asarray(y_scores).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)

        metrics = performance(y_scores, y_true)
        str1 = ['%s: %.6f ' % item for item in metrics.items()]
        self.logger.info('%s', '\n'.join(str1))
        self.logger.handlers[0].stream.close()
        self.logger.removeHandler(self.logger.handlers[0])

        return metrics

    def __call__(self, dataset, n_imgs, inside_FoV=False, logfile_path = './performance.log'):
        self.logfile_path = logfile_path
        self.init_logger(self.logfile_path)
        self.lst_predictions = []
        dirname = os.path.dirname(logfile_path)
        for i in range(n_imgs):
            img = dataset.lst_imgs[i]
            fname_img = dataset.lst_img_filenames[i]
            fname, fname_ext = os.path.splitext(fname_img)
            # Transform to tensor
            img = TF.to_tensor(img)
            if hasattr(dataset.__class__, 'lst_fovs'):
                fov = dataset.lst_fovs[i, :, :, 0]
                predict_img = self.predict_img(img) * fov
            else:
                predict_img = self.predict_img(img)
            print(dirname + '/' + fname+'.png')
            cv2.imwrite(dirname + '/' + fname+'.png', (predict_img*255).astype(np.uint8))
            self.lst_predictions.append(predict_img)
        self.lst_predictions = np.asarray(self.lst_predictions, dtype=np.float64)
        self.eval_performance(dataset, inside_FoV=inside_FoV)
