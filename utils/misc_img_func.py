import numpy as np
import cv2


def paste_imgs(lst_imgs, n_rows, n_cols, sep=5):
    n_imgs = len(lst_imgs)

    assert (n_imgs <= n_rows * n_cols)
    h, w = lst_imgs[0].shape[:2]

    new_im = np.ones(((h+sep)*n_rows+sep, (w+sep)*n_cols+sep,3))*255

    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            img = lst_imgs[k]
            if len(img.shape)==2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            row = (h+sep)*i+sep
            col = (w+sep)*j+sep
            new_im[row:row+h, col:col+w] = img
            k+=1
            if k>=n_imgs:
                break
    if sep>0:
        return new_im[sep:-sep,sep:-sep]
    return new_im
