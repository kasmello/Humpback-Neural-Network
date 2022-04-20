'''
this is where I will write all my classes :)
'''

import os
import torch
import torchvision
import glob
import random
import shutil
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from math import ceil
from torchvision import transforms, datasets
from PIL import Image, ImageDraw
from scipy import signal
from scipy import ndimage
from scipy.io import wavfile
from pydub import AudioSegment

def wav_to_spectogram(item, save = True):
    fs, x = wavfile.read(item)
    Pxx, freqs, bins, im = plt.specgram(x, Fs=fs,NFFT=1024)
    # plt.pcolormesh(bins, freqs, 10*np.log10(Pxx))
    plt.imshow(10*np.log10(Pxx), cmap='gray_r')
    plt.axis('off')
    plt.show()
    if save:
        plt.savefig(f'{item[:-4]}.png',bbox_inches=0)
        print('saved!')

class nn_data:

    def __init__(self, root):
        self.all_labels = self.make_folders(root)
        self.all_labels.sort()
        self.enum_labels = self.enum_lbl()
        self.train_path, self.validation_path = self.stratify_sample(root)
        self.all_training, self.all_validation = self.grab_dataset()

    def enum_lbl(self):
        d = {}
        for i in range(len(self.all_labels)):
            d[self.all_labels[i]] = i
        return d

    def inverse_encode(self, llist):
        return [self.all_labels[item] for item in llist]

    def grab_dataset(self):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        train_folder = datasets.ImageFolder(self.train_path,transform=transform)
        all_training = torch.utils.data.DataLoader(train_folder,
                                              batch_size=50,
                                              shuffle=True,
                                              num_workers=4)

        validation_folder = datasets.ImageFolder(self.validation_path,transform=transform)
        all_validation = torch.utils.data.DataLoader(validation_folder,
                                              batch_size=50,
                                              num_workers=4)
        return all_training, all_validation


    def pad_all_spectograms(self,pad_ms=2200):
        for item in self.wavs:
            audio = AudioSegment.from_wav(item)
            if len(audio) != 2200:
                print(f'original length: {len(audio)}')
                silence1 = AudioSegment.silent(duration=int((pad_ms-len(audio))/2))
                silence2 = AudioSegment.silent(duration=ceil((pad_ms-len(audio))/2))
                new_length = len(audio)+len(silence1)+len(silence2)
                print(f'new length: {new_length}')
                padded = silence1 + audio + silence2  # Adding silence after the audio
                padded.export(item, format='wav')

    def make_folders(self, root):
        all_dir = [dir for root, dir, file in os.walk(root)]

        all_dir = all_dir[0]
        try:
            os.mkdir(os.path.join(root,'Training'))
        except FileExistsError:
            all_dir.remove('Training')
        try:
            os.mkdir(os.path.join(root,'Validation'))
        except FileExistsError:
            all_dir.remove('Validation')
        print(all_dir)
        return all_dir

    def stratify_sample(self, root):
        random.seed(30)
        for dir in self.all_labels:
            dir_ec = str(self.enum_labels[dir])
            try:
                os.mkdir(os.path.join(root,'Training',dir_ec))
            except FileExistsError:
                print('folder exists: carry on')
            try:
                os.mkdir(os.path.join(root,'Validation',dir_ec))
            except FileExistsError:
                print('folder exists: carry on')
            label_dir = os.path.join(root,dir)
            all_files = [file for root, dir, file in os.walk(label_dir)]
            all_files = all_files[0]
            all_files = [file for file in all_files if file[-3:]=='png']
            random.seed(30)
            samp_length = int(len(all_files)*0.25)
            samp = random.sample(all_files,samp_length)
            for file in tqdm(all_files):
                if file in samp:
                    old_dir = os.path.join(root,dir,file)
                    new_dir = os.path.join(root,'Validation',dir_ec,file)
                    shutil.move(old_dir,new_dir)
                else:
                    old_dir = os.path.join(root,dir,file)
                    new_dir = os.path.join(root,'Training',dir_ec,file)
                    shutil.move(old_dir,new_dir)
        return os.path.join(root, 'Training'), os.path.join(root, 'Validation')

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        temp = torch.randn(220,220).view(-1,1,220,220)
        self._to_linear = None
        self.convs(temp)

        self.fc1 = nn.Linear(self._to_linear,512)
        self.fc2 = nn.Linear(512,23)

    def convs(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(3,3)) #3 by 3 pooling
        x = F.max_pool2d(F.relu(self.conv2(x)),(3,3))
        x = F.max_pool2d(F.relu(self.conv3(x)),(3,3))

        if not self._to_linear:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(220*220,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,1024)
        self.fc4 = nn.Linear(1024,23)

    def forward(self,x):
        x = F.relu(self.fc1(x)) #F.relu is an activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)#probability distribution

class PatchEmbed(nn.Module):
    """ Split image into patches (like a jigsaw puzzle)

    Params:

    img_size: int
        size of image

    patch_size: int
        size of patch

    in_chans: int
        number of input channels

    embed_dim: int
        the embedding dimension

    Attributes:

    n_patches: int
        number of patches inside image

    proj: nn.Conv2d
        convolutional layer that does both splitting and embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3,embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
                    in_chans,
                    embed_dim,
                    kernel_size=patch_size, #so kernels don't overlap
                    stride=patch_size,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches, embed_dim)`.
        """
        x = self.proj(
                x
            )  # (n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
            #this is a 4 dimensional tensor, about to be flattened
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        return x

class Attention(nn.Module):
    """Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    attn_p : floatk
        Dropout probability applied to the query, key and value tensors.
    proj_p : float
        Dropout probability applied to the output tensor OVERFITTING PURPOSES.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
        """

    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) #linear mapping
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)


    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError #sanity check!

                    #qkv = queries, keys, values
        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = ( #dot product
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x

class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """

    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features) ACTIVATION
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features) DROPOUT
        x = self.fc2(x)  # (n_samples, n_patches + 1, out_features) LINEAR
        x = self.drop(x)  # (n_samples, n_patches + 1, out_features) DROPOUT

        return x

class Block(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) #normalisation module, 0.000001 is just to match pretrained model
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x)) #residual layer
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision transformer.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    depth : int
        Number of blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    p, attn_p : float
        Dropout probability.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            img_size=384,
            patch_size=16,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(  # this is the transformer encoder
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x) # run through all the blocks

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x
