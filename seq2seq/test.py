
"""
色々試すだけの特に意味のないモジュール
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle

import csv, re
from gensim.corpora import Dictionary


with open("security_blog.txt", encoding="utf-8") as fp:
    text = fp.read()
    text = re.sub(r'[-“”’,–!?:;^(^)^[^]', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = text.lower()
    text = text.split('. ')
    text = [x for x in text if x]

with open("security_blog.csv", "w", encoding="utf-8", newline="") as fp:
    writer = csv.writer(fp, delimiter=",")
    for i in range(len(text)//2):
        writer.writerow([text[i], text[i+1]])
