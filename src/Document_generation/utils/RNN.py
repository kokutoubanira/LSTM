import torch
import torch.nn as nn
import torch.nn.functional as F


# class RNN(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim=50, hidden_dim=50, num_layers=1, dropout=0.2):
#         super().__init__()
#         self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
#         self.linear=nn.linear(hidden_size, 1)

#     def forward(self, x, h0 =None, l=Noen):
#         #IDをEmbeddingで多次元ベクトルに変換する
#         #xは(batch_size, step_size)
#         #->(batch_size, step_size, embedding_dim)
#         x = self.emb(x)
#         #初期状態h0とともにRNNにxを渡す
#         #xは(batch_size, step_size, embedding_dim)
#         x, h = self.lstm(x, h0)
#         #最後のステップのみ取り出す
#         if l is not None:
#             #入力のもともとの長さがある場合はそれを使用する
#             x = x[list(range(len(x))), l-1,:]
#         else :
#             #なければ最後を使用
#             x = x[:, -1, :]
#         x = self.linear(x)
#         x = x.squeeze()
#         return x

# class MyLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.dropout1 = nn.Dropout()
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, bidirectional=True, dropout=0.5)
#         self.dropout2 = nn.Dropout()
#         self.fc = nn.Linear(hidden_dim*2, output_dim)
        
#     def init_hidden(self):
#         return (torch.zeros(1, 1, self.hidden_dim),
#                 torch.zeros(1, 1, self.hidden_dim))

#     def forward(self, sentence):
#         embedded = self.dropout1(self.embedding(x))
#         output, (hidden, cell) = self.lstm(embedded)
#         hidden = self.dropout2(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
#         return self.fc(hidden)

class GenerateNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=50, hidden_size=50, num_layers=5, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        #Linerのoutputのサイズは最初のembedingのinputと同じnum_embeddings
        self.linear = nn.Linear(hidden_size, num_embeddings)

    def forward(self, x, h0=None):
        x = self.emb(x)
        x,h = self.lstm(x, h0)
        x = self.linear(x)

        return x, h
