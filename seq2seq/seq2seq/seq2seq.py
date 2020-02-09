"""
seq2seq の Pytorch 実装
"""

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        word2id: dict,
        batch_size: int = 128

    ) -> None:
        """
        :param vocab_size: 単語index数
        :param embedding_dim: 埋め込み層の次元
        :param hidden_dim: 隠れ層の次元
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
        input(indices): [batch_size, seq_len]
        embedding_out: [batch_size, seq_len, embedding_dim]
        hidden: [num_layers * num_directions, batch_size, hidden_dim]
        gru_hidden_out(state): [num_layers * num_directions, batch_size, hidden_dim]

        :param indices: 単語の index を含んだテンソル
        Comment: num_layers * num_directions はレイヤ*方向(一方向か双方向か)の値
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
        word2id: dict,
        batch_size: int = 128
        
    ) -> None:
        """
        :param vocab_size: 単語index数
        :param embedding_dim: 埋め込み層の次元
        :param hidden_dim: 隠れ層の次元
        :param batch_size: ミニバッチサイズ
        """
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
        単語列 x_(0) ~ x_(t-1) から単語列 x_(1) ~ x_(t) を予測

        input: [batch_size, seq_len]
        hidden_in: [num_layers * num_directions, batch_size, hidden_dim]
        embedding_out: [batch_size, seq_len, embedding_dim]
        gru_hidden_out: [num_layers * num_directions, batch_size, hidden_dim]
        gru_out: [batch_size, seq_len, hidden_dim]
        output: [batch_size, seq_len, vocab_size]

        :param indices: torch.tensor
        :param state: torch.tensor
        """
        embedding = self.word_embeddings(indices)
        if embedding.dim() == 2:
            embedding = torch.unsqueeze(embedding, 1)
        gruout, state = self.gru(embedding, state)
        output = self.output(gruout)
        return output, state



