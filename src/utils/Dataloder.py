import MeCab
import re
from sklearn.model_selection import train_test_split
import random
from sklearn.utils import shuffle
import torchtext
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
    sentence = tagger.parse(sentence)
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    wakati = sentence.split(" ")
    wakati = list(filter(("").__ne__, wakati))
    return wakati

word2index = {}
# 系列を揃えるためのパディング文字列<pad>を追加
# パディング文字列のIDは0とする
word2index.update({"<pad>":0})

for title in datasets["title"]:
    wakati = make_wakati(title)
    for word in wakati:
        if word in word2index: continue
        word2index[word] = len(word2index)
print("vocab size : ", len(word2index))


class MDatasets(Dataset):
    def __init__(self, data):
        