import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import csv
from gensim.corpora import Dictionary

from seq2seq.seq2seq import Encoder, Decoder


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus = []
with open('dataset.csv', encoding='utf-8') as fp:
    reader = csv.reader(fp)
    for i, row in enumerate(reader):
        if i == 0: pass
        corpus.append(row[0].split(' '))
        corpus.append(row[1].split(' '))
N = len(corpus) // 2
dct = Dictionary(corpus)
word2id = dct.token2id
initialize = dct[0]
dct_len = len(word2id)
word2id.update({"<pad>": dct_len, "<eos>": dct_len+1})
id2word = {v: k for k, v in word2id.items()}

seq_len = 10
def load_dataset():
    def load_sent_list(training=True):
        sent_list = []
        with open('dataset.csv', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            for i, row in enumerate(reader):
                if i == 0: pass
                if training:
                    sent_list.append(row[0].split(' '))
                else:
                    sent_list.append(row[1].split(' '))
        return sent_list


    def padding(sent_list, seq_len=seq_len):
        res = []
        for i, sent in enumerate(sent_list):
            if len(sent) > seq_len:
                res.append(sent[:seq_len])
            else:
                for j in range(seq_len - len(sent)):
                    sent.append("<pad>")
                res.append(sent)
        return res


    def transform(sent_list):
        res = []
        for sent in sent_list:
            tmp = []
            for i, word in enumerate(sent):
                try:
                    tmp.append(word2id[word])
                except:
                    tmp.append(word2id["<pad>"])
            res.append(tmp)
        return res
    

    source = []
    target = []
    for _ in range(N):
        eos = word2id["<eos>"]
        x = load_sent_list(training=True)
        y = load_sent_list(training=False)
        left = padding(x)
        right = padding(y)
        source.append(transform(left))
        right = transform(right)
        right = [[eos] + right[i][:seq_len-1] for i, _ in enumerate(right)]
        for i, _ in enumerate(right):
            right[i][right[i].index(word2id["<pad>"])] = eos
        target.append(right)
    
    return source, target

source, target = load_dataset()
train_x, test_x, train_t, test_t = train_test_split(source, target, test_size=0.1)


def train2batch(source, target, batch_size=128):
    input_batch = []
    output_batch = []
    source, target = shuffle(source, target)

    for i in range(0, len(source), batch_size):
        input_tmp = []
        output_tmp = []
        for j in range(i, i + batch_size):
            input_tmp.append(source[j])  # append([vocab_size, idx])
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
    batch_size = 2

    # 計算グラフを定義
    # (1) ネットワークをインスタンス化し，推論グラフを定義
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, word2id).to(device)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, word2id).to(device)

    # (2) 損失を生成するグラフを定義する
    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"])

    # (3) 勾配を計算し適用する操作を定義する
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    # 学習フェーズ
    print("Training...")
    n_epoch = 100
    for epoch in range(1, n_epoch + 1):

        input_batch, output_batch = train2batch(train_x, train_t, batch_size=batch_size)
        for i, _ in enumerate(input_batch):
            # Zero gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # Prepare tensor
            inputs = torch.tensor(input_batch[i], device=device)  # [batch_size, vocab_size, 1]
            outputs = torch.tensor(output_batch[i], device=device)  # [batch_size, vocab_size, 1]
            print(inputs.shape)
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
    main()