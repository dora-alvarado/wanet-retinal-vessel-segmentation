from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d
from kornia.geometry.transform import rotate, remap
#from kornia import remap
from scipy.ndimage.filters import gaussian_filter
import numpy as np


def edge_conv2d(im, in_ch, out_ch):
    # Use nn.Conv2d to define the convolution operation
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # Define the sobel operator parameters
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float)
    # Convert the sobel operator into a convolution kernel adapted to the convolution operation
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # Convolution output channel, here I set it to 3
    sobel_kernel = np.repeat(sobel_kernel, in_ch, axis=1)
    # Enter the channel of the picture, here I set it to 3
    sobel_kernel = np.repeat(sobel_kernel, out_ch, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).type(im.type())
    # print(conv_op.weight.size())
    # print(conv_op, '\n')
    #print(im.shape)
    edge_detect = conv_op(im.reshape(1,1, im.shape[0], im.shape[1]))
    #print(torch.max(edge_detect))
    # Convert output to image format
    edge_detect = edge_detect.squeeze()#.detach().numpy()
    return edge_detect


def gaussian1d(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / (2. * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(gauss_fcn(x)) for x in range(window_size)])
    return gauss / gauss.sum()


def gmf(window_size, sigma, order=0):
    wsize_x, wsize_y = window_size
    row = gaussian1d(wsize_x, sigma).view(1, -1)
    f = row.repeat(wsize_y, 1)
    for i in range(order):
        f = edge_conv2d(f, 1, 1)
    return f #/ f.sum()


def get_gmf_kernel(kernel_size: Tuple[int, int],
                   sigma: float, order = 0) -> torch.Tensor:
    r"""
    Function that returns GMF filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 2D tensor with GMF filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> filters.get_gmf_kernel((3, 3), 1.)
        tensor([[0.0914, 0.1506, 0.0914],
        [0.0914, 0.1506, 0.0914],
        [0.0914, 0.1506, 0.0914]])

    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}"
                        .format(kernel_size))
    kernel_2d: torch.Tensor = gmf(kernel_size, sigma, order=order)
    return kernel_2d


class ApplyGMF(nn.Module):
    r"""Creates an operator that applies a GMF to a tensor.

    The operator enhances curvilinear structures from the given tensor
    with a GMF kernel by convolving  it to each channel.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma float: the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = filters.ApplyGMF((3, 3), 1.5)
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigmas: torch.Tensor,
                 n_orientations: int = 12) -> None:
        super(ApplyGMF, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigmas: torch.Tensor = sigmas
        self.n_orientations: int = n_orientations
        self._padding: Tuple[int, int] = self.compute_padding(kernel_size)
        self.kernels = None
        self.n_filters =len(sigmas)

    @staticmethod
    def compute_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        kernels = []
        x_padding = F.pad(input=x, pad=self._padding + self._padding, mode='reflect', value=0)
        for s in self.sigmas:
            kernel: torch.tensor = get_gmf_kernel(self.kernel_size, s)
            kernel = kernel.repeat(1, c, 1, 1)
            kernels.append(kernel)
        self.kernels = torch.cat(kernels, dim=0)
        responses = []
        for theta in range(self.n_orientations):
            angle = theta * (180 / self.n_orientations)  # / self.n_orientations * pi
            angle = torch.tensor([angle], requires_grad=False).to(x.device)
            rkernel = rotate(self.kernels, angle, align_corners=True)
            response = conv2d(x_padding, rkernel - rkernel.mean(), padding=self._padding, stride=1)
            responses.append(response)
        all = torch.stack(responses)
        max_responses, indices = all.max(0)

        return max_responses


class ApplyDGMF(nn.Module):
    r"""Creates an operator that applies a D-GMF to a tensor.

    The operator enhances curvilinear structures from the given tensor
    with a GMF kernel by convolving  it to each channel.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma float: the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> dgauss = filters.ApplyDGMF((3, 3), 1.5)
        >>> output = dgauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigmas: torch.Tensor,
                 alphas: torch.Tensor,
                 n_orientations: int = 24,
                 dx=None, dy=None,
                 order = 0) -> None:
        super(ApplyDGMF, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigmas: torch.Tensor = sigmas
        self.alphas: torch.Tensor = alphas
        self.n_orientations: int = n_orientations
        self._padding: Tuple[int, int] = self.compute_padding(kernel_size)
        self.dx = dx
        self.dy = dy
        self.kernels = None
        self.n_filters = len(sigmas)
        self.responses = None
        self.order = order

    @staticmethod
    def compute_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        b, c, h, w = x.shape

        if self.dx is None:
            self.dx = torch.tensor(
                gaussian_filter((torch.rand(*self.kernel_size) * 2 - 1), 4., mode="constant", cval=0)).to(
                x.device, dtype=x.dtype)
            self.dx = self.dx.repeat(1, 1, 1)
        if self.dy is None:
            self.dy = torch.tensor(
                gaussian_filter((torch.rand(*self.kernel_size) * 2 - 1), 4., mode="constant", cval=0)).to(
                x.device, dtype=x.dtype)
            self.dy = self.dy.repeat(1, 1, 1)

        # prepare kernel
        idx_x = torch.arange(self.kernel_size[1], dtype=x.dtype)
        idx_y = torch.arange(self.kernel_size[0], dtype=x.dtype)
        grid_x, grid_y = torch.meshgrid(idx_x, idx_y)
        grid_x = grid_x.repeat(1, 1, 1).to(x.device)
        grid_y = grid_y.repeat(1, 1, 1).to(x.device)

        # convolve tensor with kernel
        kernels = []
        x_padding = F.pad(input=x, pad=self._padding + self._padding, mode='reflect', value=0)
        for i, s in enumerate(self.sigmas):
            # get GMF
            kernel: torch.tensor = get_gmf_kernel(self.kernel_size, s, order=self.order).to(x.dtype)
            kernel = kernel.repeat(1, c, 1, 1)
            # get D-GMF (distorted filter)
            kernel = remap(kernel, grid_x + self.dx * 10 * self.alphas[i],
                           grid_y + self.dy * 10 * self.alphas[i])
            kernels.append(kernel)
        self.kernels = torch.cat(kernels, dim=0)

        responses = []
        for theta in range(self.n_orientations):
            angle = theta * (360 / self.n_orientations)
            angle = torch.tensor([angle], requires_grad=False).to(x.device)
            rkernel = rotate(self.kernels, angle, align_corners=True)
            response = conv2d(x_padding, rkernel - rkernel.mean(), padding=0, stride=1)
            responses.append(response)
        all = torch.stack(responses)
        max_responses, indices = all.max(0)
        self.responses = max_responses
        return max_responses


"""
#how to use:

import matplotlib.pyplot as plt

torch.manual_seed(0)
# filter size
n = 7
# input in format BxCxHxW
dummy = torch.rand(1, 3, 15, 15)
b, c, h, w = dummy.shape
# set sigmas and alphas values
sigmas = torch.tensor([0.5]).double()
alphas = torch.tensor([3.]).double()
# create filters
dgauss = ApplyDGMF((n, n), sigmas, alphas)
# apply filters
output = dgauss(dummy)  # 1x3x15x15
# plot first output 
plt.imshow(output[0, 0, :, :], cmap='gray')
plt.show()
# plot first filter
plt.imshow(dgauss.kernels[0, 0], cmap='gray')
plt.show()

"""