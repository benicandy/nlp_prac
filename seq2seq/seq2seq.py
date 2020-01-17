"""
seq2seq の Pytorch 実装
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        batch_size: int = 128
    ) -> None:
        """
        :param vocab_size: 単語列の長さ
        :param embedding_dim: 埋め込み次元
        :param hidden_dim: 隠れ層次元
        :param batch_size: ミニバッチサイズ
        """
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

    def forward(self,
        indices: torch.tensor
    ) -> torch.tensor:
        """
        input: [batch_size, vocab_size, 1]
        embedding_out: [batch_size, vocab_size, embedding_dim]
        hidden: [1, batch_size, hidden_dim]
        gru_hidden_out: [1, batch_size, hidden_dim]

        :param indices: 単語の index を含んだテンソル
        Comment: hidden や gru_hidden_out の 1 はレイヤ*方向の値
        """
        embedding = self.word_embeddings(indices)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        _, state = self.gru(embedding, torch.zeros(1, self.batch_size, self.hidden_dim, device=device))

        return state


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        batch_size: int = 128
    ) -> None:
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2id["<pad>"])
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        indices: int,
        state: torch.tensor,
    ) -> torch.tensor:
        """
        input: [batch_size, vocab_size, 1]
        hidden_in: [1, batch_size, hidden_dim]
        embedding_out: [batch_size, vocab_size, embedding_dim]
        gru_hidden_out: [1, batch_size, hidden_dim]
        gru_out: [batch_size, vocab_size, hidden_dim]
        output: [batch_size]

        :param indices: torch.tensor
        :param state: torch.tensor
        """
        embedding = self.word_embeddings(indices)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        gruout, state = self.gru(embedding, state, device=device)
        output = self.output(gruout)

        return output, state

