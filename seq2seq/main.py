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

from seq2seq.seq2seq import Encoder, Decoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus = []
with open('dataset.csv', encoding='utf-8') as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        print(row)
        if i == 0: pass
        corpus.append(row[0].split(' '))
        corpus.append(row[1].split(' '))
dct = Dictionary(corpus)
word2id = dct.token2id
print("initialize: ", dct[0])

dct_len = len(word2id)
word2id.update({"<pad>": dct_len, "<eos>": dct_len+1})
id2word = {v: k for k, v in word2id.items()}

print(word2id)
print(id2word)


def load_dataset():
    def load_sent():
        with open('dataset.csv', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0: pass



    def padding(string, training=True):
        pass

    def transform(string, seq_len=10):
        pass


def train2batch(data, target, batch_size=128):
    input_batch = []
    output_batch = []
    data, target = shuffle(data, target)

    for i in range(0, len(data), batch_size):
        input_tmp = []
        output_tmp = []
        for j in range(i, i + batch_size):
            input_tmp.append(data[j])  # append([vocab_size, idx])
            output_tmp.append(target[j])  # append([vocab_size, idx])
        input_batch.append(input_tmp)  # append([batch_size, vocab_size, idx])
        output_batch.append(output_tmp)  # append([batch_size, vocab_size, idx])
    return input_batch, output_batch  # [batch, batch_size, vocab_size, idx]

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    
    embedding_dim = 100
    hidden_dim = 128
    vocab_size = len(word2id)
    batch_size = 128

    # 計算グラフを定義
    # (1) ネットワークをインスタンス化し，推論グラフを定義
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim).to(device)

    # (2) 損失を生成するグラフを定義する
    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"])

    # (3) 勾配を計算し適用する操作を定義する
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    # 学習フェーズ
    print("Training...")
    n_epoch = 100
    for epoch in range(1, n_epoch + 1):

        input_batch, output_batch = train2batch(train_x, train_t)
        for i in range(len(input_batch)):
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # Prepare tensor
            inputs = torch.tensor(input_batch[i], device=device)  # [batch_size, vocab_size, 1]
            outputs = torch.tensor(output_batch[i], device=device)  # [batch_size, vocab_size, 1]
            # Forward pass through encoder
            encoder_hidden = encoder(inputs)
            # Create source and target
            source = outputs[:, :-1]  # [batch_size, vocab_size-1, 1]
            target = outputs[:, 1:]  # [batch_size, vocab_size-1, 1]
            decoder_hidden = encoder_hidden
            # Forward batch of sequences through decoder one time step at a time
            loss = 0
            for i in range(source.size(1)):
                decoder_output, decoder_hidden = decoder(source[:, i], decoder_hidden)
                decoder_output = torch.squeeze(decoder_output)
                loss += criterion(decoder_output, target[:, i])
            
            # 損失を逆伝播
            loss.backward()

            # パラメータを更新
            encoder_optimizer.step()
            decoder_optimizer.step()
        
        if epoch % 10 == 0:
            print(get_current_time(), "Epoch %d: %.2f" % (epoch, loss.item()))
        
        if epoch % 10 == 0:
            model_name = "seq2seq_calculator_v{}.pt".format(epoch)
            torch.save({
                'encoder_model': encoder.state_dict(),
                'decoder_model': decoder.state_dict(),
            }, model_name)
            print("Saving the checkpoint...")


if __name__ == "__main__":
    pass