import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle

import csv
from gensim.corpora import Dictionary


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

