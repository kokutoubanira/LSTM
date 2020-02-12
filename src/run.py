from sklearn.model_selection import train_test_split
import torch.optim as optim
import LSTM.LSTM_NN as NN
import torch
import torch.nn as nn 
import torch.nn.functional as F
import train



# 元データを7:3に分ける（7->学習、3->テスト）
traindata, testdata = train_test_split(datasets, train_size=0.7)

# 単語のベクトル次元数
EMBEDDING_DIM = 10
# 隠れ層の次元数
HIDDEN_DIM = 128
# データ全体の単語数
VOCAB_SIZE = len(word2index)
# 分類先のカテゴリの数
TAG_SIZE = len(categories)

from torchtext.vocab import FastText
TEXT.build_vocab(train, vectors=FastText(language="ja"), min_freq=2)












model = NN(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAG_SIZE)
loss_function = nn.NLLLoss()

opt = optim.Adam(model.parameters(), lr=0.01)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": traindata, "val": testdata}

