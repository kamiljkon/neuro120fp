#---PACKAGE IMPORT ---
# Comp. Neuro specific libraries
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import ipywidgets as widgets
from IPython.display import display
from torchcam.methods import GradCAM
from monai.visualize.utils import blend_images, matshow3d
import scipy.ndimage as ndi 
from nilearn import datasets, plotting, image, surface 
from nilearn.plotting import plot_surf_contours, plot_surf
import umap

# General libraries
import numpy as np
import pandas as pd
import glob
from collections import defaultdict, Counter
from nilearn.image import resample_to_img
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import math
import pickle
import multiprocessing
import gc

# ML libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from monai.networks.nets import resnet
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

#--- The 3D CNN model ---
class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=1):
        super().__init__()
        hidden_dim = in_ch * expansion
        self.expand = nn.Conv3d(in_ch, hidden_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1,
                                groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        self.project = nn.Conv3d(hidden_dim, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = F.relu6(self.bn1(self.expand(x)))
        out = F.relu6(self.bn2(self.dwconv(out)))
        out = self.bn3(self.project(out))
        return x + out if self.use_res_connect else out

class Bigger3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=5, dropout_p=0.3):
        super().__init__()
        # initial conv wider
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(32)

        # separable blocks with increased width
        self.ds1 = DepthwiseSeparableConv3D(32, 64, stride=2, expansion=2)
        self.ds2 = DepthwiseSeparableConv3D(64, 128, stride=2, expansion=2)
        self.ds3 = DepthwiseSeparableConv3D(128, 256, stride=2, expansion=2)
        # an extra block
        self.ds4 = DepthwiseSeparableConv3D(256, 256, stride=2, expansion=2)

        self.pool    = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(dropout_p)
        self.fc      = nn.Linear(256, num_classes)
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.ds1(x); x = self.ds2(x)
        x = self.ds3(x); x = self.ds4(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
