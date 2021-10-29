import torch
import torch.nn.functional as F


def pad_overlap(img, patch_size, stride):
    # Calculate padding to fit the sliding windows
    pad1 = stride - (img.size(1) - patch_size) % stride  # leftover on the h dim
    pad2 = stride - (img.size(2) - patch_size) % stride  # leftover on the w dim
    img = F.pad(img, (0, pad2, 0, pad1, 0, 0))
    return img


def unfold(img, patch_size, stride):
    c, h, w = img.shape
    patches = img.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.reshape((-1, c, patch_size, patch_size))
    return patches


def patches_overlap (test_img, patch_size, stride):
    test_img = pad_overlap(test_img, patch_size, stride)
    patches_img_test = unfold(test_img, patch_size, stride)
    return patches_img_test, test_img.shape[1], test_img.shape[2]


def fold(pred_patches, img_c, img_h, img_w, stride):
    # https://stackoverflow.com/questions/62995726/pytorch-sliding-window-with-unfold-fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold
    # input tensor must be of shape (b, c*(patch_size*patch_size), l), where l is the total number of patches
    # b, c, l, patch_size*patch_size -->
    b = 1
    patch_size = pred_patches.shape[-1]
    patches_reverse = torch.from_numpy(pred_patches).reshape((1, img_c, -1, patch_size**2))
    # b, c, patch_size*patch_size, l -->
    patches_reverse = patches_reverse.permute(0, 1, 3, 2)
    # b, c*patch_size*patch_size, l -->
    patches_reverse = patches_reverse.view(b, img_c*patch_size**2, -1)
    output = F.fold(patches_reverse, output_size=(img_h, img_w), kernel_size=patch_size, stride=stride)
    recovery_mask = F.fold(torch.ones_like(patches_reverse), output_size=(img_h, img_w), kernel_size=patch_size, stride=stride)
    output = output / recovery_mask
    return output.numpy()


def recompone_overlap(pred_patches, img_h, img_w, stride):
    n_patches, n_c, patch_h, patch_w = pred_patches.shape
    N_patches_h = (img_h - patch_h) // stride + 1
    N_patches_w = (img_w - patch_w) // stride + 1
    N_patches_img = N_patches_h * N_patches_w
    assert (pred_patches.shape[0] % N_patches_img == 0)
    final_avg = fold(pred_patches, n_c, img_h, img_w, stride)
    return final_avg
