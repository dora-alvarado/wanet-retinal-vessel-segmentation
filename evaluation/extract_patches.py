import torch.nn.functional as F
import numpy as np

def pad_overlap(img, patch_size, stride):
    # Calculate padding to fit the sliding windows
    pad1 = stride - (img.size(1) - patch_size) % stride  # leftover on the h dim
    pad2 = stride - (img.size(2) - patch_size) % stride  # leftover on the w dim
    img = F.pad(img, (0, pad2, 0, pad1, 0, 0))
    return img


def unfold(img, patch_size, stride):
    patches = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.reshape((-1, 1, patch_size, patch_size))
    return patches


def patches_overlap (test_img, patch_size, stride):
    test_img = pad_overlap(test_img, patch_size, stride)
    patches_img_test = unfold(test_img, patch_size, stride)
    return patches_img_test, test_img.shape[1], test_img.shape[2]


def recompone_overlap(pred_patches, img_h, img_w, stride):
    patch_h = pred_patches.shape[2]
    patch_w = pred_patches.shape[3]
    N_patches_h = (img_h - patch_h) // stride + 1
    N_patches_w = (img_w - patch_w) // stride + 1
    N_patches_img = N_patches_h * N_patches_w

    assert (pred_patches.shape[0] % N_patches_img == 0)
    N_full_imgs = pred_patches.shape[0] // N_patches_img

    full_prob = np.zeros((N_full_imgs, pred_patches.shape[1], img_h, img_w))
    full_sum = np.zeros((N_full_imgs, pred_patches.shape[1], img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride + 1):
            for w in range((img_w - patch_w) // stride + 1):
                full_prob[i, :, h * stride:(h * stride) + patch_h, w * stride:(w * stride) + patch_w] += \
                pred_patches[k]
                full_sum[i, :, h * stride:(h * stride) + patch_h, w * stride:(w * stride) + patch_w] += 1
                k += 1
    assert (k == pred_patches.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    assert (np.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
    assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
    return final_avg
#assert(np.max(test_mask)==1  and np.min(test_mask)==0)
