import numbers
import random
import warnings
import torch
import numpy as np
import torch.nn as nn
from collections.abc import Sequence
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
warnings.filterwarnings("ignore", category=UserWarning) 
tf.config.experimental.set_visible_devices([], 'GPU')

# __all__ just exports the below things if the whole module is called xD
__all__ = [ 
    "TimeWarp",
    "FreqMask",
    "TimeMask",
    "normalise"
]

class TimeWarp(nn.Module):
    def __init__(self,p=0.2, T=70):
        """
        input:
            p: probability of warping
            W: number of pixels function is capable of warping
        """
        super().__init__()
        self.p = p
        self.T = T
    #steps
    #1, get height + length
    # height //2, assign to y
    #assign horizontal line with img[0][y]
    #pick a random point to warp from W to lenth-W
    #choose how far to warp it from -w to w
    def forward(self, img):
        if random.random() > self.p:
            return img
        img = np.reshape(img, (-1, img.shape[1],img.shape[2],1))
        height, time = img.shape[1], img.shape[2]
        y = height // 2
        horizontal_line_through_center = img[0][y]
        assert len(horizontal_line_through_center) == time
        point_to_warp = horizontal_line_through_center[random.randrange(self.T, time - self.T)]
        assert isinstance(point_to_warp, torch.Tensor)
        dist_to_warp = random.randrange(-self.T, self.T)
        src_pts, dest_pts = torch.tensor([[[y, point_to_warp]]]), torch.tensor([[[y, point_to_warp + dist_to_warp]]])
        warped_spectro, _ = sparse_image_warp(img, src_pts, dest_pts,num_boundary_points=2)
        return torch.tensor(np.reshape(warped_spectro, (-1, warped_spectro.shape[1],warped_spectro.shape[2])))

    def __repr__(self) -> str: #representation of the object so that it is not unambiguous
        return f"{self.__class__.__name__}(p={self.p})(W={self.T})"

class FreqMask(nn.Module):
    def __init__(self, p = 0.2, F=20, masks=5):
        super().__init__()
        self.p = p
        self.F = F
        self.masks = masks

    def forward(self, img):
        if random.random() > self.p:
            return img
        num_mel_channels = img.shape[1]

        for i in range(0, self.masks):
            f = random.randint(0, self.F)
            f_zero = random.randint(0, num_mel_channels - f)
            # avoids randrange error if values are equal and range is empty
            if (f==0): continue
            img[0][f_zero:f_zero+f] = 0 #this is the masking part
        return img


    def __repr__(self) -> str: #representation of the object so that it is not unambiguous
        return f"{self.__class__.__name__}(p={self.p})(F={self.F})(Masks={self.masks})"

class TimeMask(nn.Module):
    def __init__(self, p=0.2, T=20, masks=1):
        super().__init__()
        self.p = p
        self.T = T
        self.masks = masks

    def forward(self, img):
        if random.random() > self.p:
            return img
        len_spectro = img.shape[2]

        for i in range(0, self.masks):
            t = random.randint(0, self.T)
            t_zero = random.randint(0, len_spectro - t)
            # avoids randrange error if values are equal and range is empty
            if (t==0): continue
            img[0][:, t_zero:t_zero+t] = 0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})(T={self.T})(Masks={self.masks})"

def normalise(img):
    img = 10 * np.log10(img)
    # img = Pxx
    img = np.flipud(img)
    max_box = img.max()
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > max_box-5:
                img[i][j] = max_box - 5
            elif img[i][j] < max_box-30:
                img[i][j] = max_box-30
    img = img - img.min()
    img = img / img.max()
    return img.astype('float32')
        