import torch

x = torch.randint(0, 9, (3, 4, 1, 5, 1))
print(x.shape)

"""
指定した dim の要素数が1なら削除する
torch.squeeze(input, dim=None, out=None) → Tensor
"""
print(x.squeeze().shape)
print(torch.squeeze(x).shape)

"""
指定した dim に要素数1のテンソルを追加する？
torch.unsqueeze(input, dim, out=None) → Tensor
"""
print(x.unsqueeze(0).shape)
print(torch.unsqueeze(x, 1).shape)

"""
torch.nn.gru
input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() for details.

h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided. If the RNN is bidirectional, num_directions should be 2, else it should be 1.

output of shape (seq_len, batch, num_directions * hidden_size): tensor containing the output features h_t from the last layer of the GRU, for each t. If a torch.nn.utils.rnn.PackedSequence has been given as the input, the output will also be a packed sequence. For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively.

Similarly, the directions can be separated in the packed case.

h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t = seq_len

Like output, the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size).
"""
