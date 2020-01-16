"""
シンプルな多層 MLP の Pytorch 実装
cf1. https://qiita.com/sudamasahiko/items/b54fed1ffe8bb6d48818
cf2. https://qiita.com/perrying/items/857df46bb6cdc3047bd8
cf3. http://aidiary.hatenablog.com/entry/20180129/1517233796
※1 pytorch 1.2.0 用にコードを一部修正
※2 GPU を使えるようにコードを一部改変
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


iris = datasets.load_iris()
# 正解ラベルを one-hot 表現に変換
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
y[np.arange(len(iris.target)), iris.target] = 1
# 教師データの作成
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)
# pytorch で扱える形式に変換
# requires_grad=False だと微分の対象にならず勾配は None が返る
# 推論グラフを定義して backward() を実行するとグラフを構築する各変数の grad に勾配が入る
x = torch.tensor(X_train, requires_grad=True, device=device, dtype=torch.float)
y = torch.tensor(y_train, device=device, dtype=torch.float)


# (1) ネットワークをインスタンス化し，推論グラフを定義する
net = Net().to(device)
print("w: ", net.fc1.weight)  # 初期値
print("b: ", net.fc1.bias)  # 初期値
# (2) 損失を生成する操作を定義する
criterion = nn.MSELoss()
# (3) 勾配を計算し適用する操作を定義する
optimizer = optim.SGD(net.parameters(), lr=0.01)

# training
for i in range(1):
    # 勾配パラメータを初期化
    optimizer.zero_grad()
    # 順伝播(NN の予測を計算)
    output = net(x)
    # 予測と正解ラベルから損失を計算
    loss = criterion(output, y)
    print("loss: ", loss)
    print("dL/dw: ", net.fc1.weight.grad)
    print("dL/db: ", net.fc1.bias.grad)
    # 損失を逆伝播(勾配を計算)
    loss.backward()
    print("loss.backward()")
    print("dL/dw: ", net.fc1.weight.grad)
    print("dL/db: ", net.fc1.bias.grad)
    # パラメータの更新
    optimizer.step()
    print("w: ", net.fc1.weight)
    print("b: ", net.fc1.bias)


# test
#outputs = net(torch.tensor(X_test, device=device, dtype=torch.float))
#_, predicted = torch.max(outputs.data, 1)
