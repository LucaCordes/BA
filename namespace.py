from JetTransformer import *
from JetClassifier import *
from preprocess import *
from Heads import *
from tables_plots_helpers import *
from helpers import *

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

import sklearn
import sklearn.metrics
from pathlib import Path
import pathlib
import numpy as np
from numpy import (sqrt,pi,abs,
                   sin,arcsin,sinh,arcsinh,
                   cos,arccos,cosh,arccosh,
                   tan,arctan,tanh,arctanh,
                   exp,log,
                   deg2rad,rad2deg,vectorize)
import pandas as pd
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from collections.abc import Iterable
from pprint import pprint
import ansi.color as color
import tabulate as __tabulate__
__tabulate__.PRESERVE_WHITESPACE = True
from tabulate import tabulate
import re
from pylatexenc.latex2text import LatexNodes2Text
import os
import uncertainties as un
from tqdm.auto import tqdm
import re
import contextlib
import datetime
import time
import copy
import json
import functools
import inspect
import scipy.integrate as integrate
import scipy.constants as c
import scipy.stats as stats
import itertools
import imageio.v2 as imageio
from io import BytesIO

plt.rcParams.update(
    {
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.labelsize": "large",
        "text.usetex": False,
        "font.size": 13,
    }
)

np.seterr(divide='ignore')