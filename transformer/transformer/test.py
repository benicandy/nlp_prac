import torch

x = torch.randint(0, 9, (3, 4, 1, 5, 1))
print(x.shape)

x_sq = x.squeeze(2)
print(x_sq.shape)

x_unsq = x.unsqueeze(1)
print(x_unsq.shape)