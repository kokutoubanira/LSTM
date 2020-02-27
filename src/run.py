from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
import torch.nn as nn 
import torch.nn.functional as F
import train
import string
from utils.Dataloder import *
from utils.RNN import GenerateNN as NN
import string
import tqdm

all_chars = string.printable
vocab_size = len(all_chars)
vocab_dict = dict((c,i) for (i,c) in enumerate(all_chars))

#dataloder作成
ds = CDataset("data/tinyshakespeare.txt")
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

def generate_seq(net, start_phrase="The King said", length=200, tmperture=0.8, device="gpu"):
    #モデルを評価モードにする
    net.eval()
    #出力の数値を格納するリスト
    result = []
    
    #開始文字列をTensorに変換
    start_tensor = torch.tensor(
        str2ints(start_phrase, vocab_dict),
        dtype=torch.int64
        ).to(device)
    #先頭にbatch次元をつける
    x0 = start_tensor.unsqueeze(0)
    #RNNにとpして出力と新しい内部状態を得る
    o, h = net(x0)
    #出力を（正規化されていない）確率に変換
    out_dist = o[:,-1].view(-1).exp()
    #確率から実際の文字のインデックスをサンプリング
    top_i = torch.multinomial(out_dist, 1)[0]
    #結果を保存
    result.append(top_i)

    #生成された結果を次々にRNNに入力していく
    for i in range(length):
        inp =torch.tensor([[top_i]], dtype=torch.int64)
        inp = inp.to(device)
        o, h = net(inp, h)
        out_dist = o.view(-1).exp()
        top_i = torch.multinomial(out_dist, 1)[0]
        result.append(top_i)

    #開始文字列と生成された文字列をまとめて返す
    return start_phrase + ints2str(result, all_chars)

from statistics import mean

net = NN(vocab_size, 50, 50, num_layers=5, dropout=0.4)
net.to("cuda:0")
opt = optim.Adam(net.parameters())
#多クラスの識別で問題なのでSoftmaxCrossEntropyLossが損失関数となる
loss_f = nn.CrossEntropyLoss()

for epoch in range(50):
    net.train()
    losses = []
    for data in tqdm.tqdm(loader):
        x = data[:,:-1]
        #yは2文字目から最後の文字まで
        y= data[:,1:]
        x = x.to("cuda")
        y = y.to("cuda")
        y_pred, _ = net(x)
        #batchとstepの軸を統合してからlossに渡す
        loss = loss_f(y_pred.view(-1, vocab_size), y.view(-1))
        net.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        #現在の損失関数と生成される文書例を表示
    print(epoch, mean(losses))
    with torch.no_grad():
        print(generate_seq(net, device="cuda"))



