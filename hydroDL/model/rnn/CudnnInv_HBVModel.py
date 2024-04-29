import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask
import csv
import numpy as np


class CudnnInv_HBVModel(torch.nn.Module):

    def __init__(self, **arg):
        pass
