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
import glob
import pathlib
from torch.utils.data import Dataset, DataLoader, TensorDataset



# remove_marks_regex = re.compile("[,\.\(\)\[\]\*:;|<.*?>")
shift_marks_regex = re.compile("([?!])")

def text2ids(text, vocab_dict):
    #!?以外の記号削除
    text = remove_marks_regex("", text)
    #!?と単語の間にスペースを挿入
    text = shift_marks_regex.sub(r"\1", text)
    tokens = text.split()
    return [vocab_dict.get(token, 0) for token in tokens]

def list2tensor(token_idexes, max_len=100, padding=True):
    if len(token_idexes) > max_len:
        token_idexes = token_idexes[:max_len]
    n_tokens = len(token_idexes)
    if padding:
        token_idexes = token_idexes \
            + [0]*(max_len -len(token_idexes))
        return torch.tensor(token_idexes, dtype=torch.int64)
    

tagger = MeCab.Tagger("-Owakati")


def make_wakati(sentence):
    sentence = tagger.parse(sentence)
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    wakati = sentence.split(" ")
    wakati = list(filter(("").__ne__, wakati))
    return wakati

# word2index = {}
# # 系列を揃えるためのパディング文字列<pad>を追加
# # パディング文字列のIDは0とする
# word2index.update({"<pad>":0})

# # for title in datasets["title"]:
#     wakati = make_wakati(title)
#     for word in wakati:
#         if word in word2index: continue
#         word2index[word] = len(word2index)
# print("vocab size : ", len(word2index))


class MYDataset(Dataset):
    def __init__(self, dir_path, trian=True, max_len=100, padding=True):
        self.max_len = max_len
        self.padding = padding
        path = pathlb.Path(dir_path)
        vocab_path = path.joinpath("*.vocab")

        #ボキャブラリファイル読み込み
        self.vocab_array = vocab_path.open() \
            .read().strip().splitlines()
        #単語をキーとし,値がidのdict
        self.vocab_dict = dict((w, i+1)\
            for i, w in enumerate(self.vocab_array))
        if train:
            target_path = path.joinpath("train")
        else:
            target_path = path.joinpath("test")
        pos_files = sorted(glob.glob(
            str(target_path.joinpath("pos/*.txt"))
        ))
        neg_files = sorted(glob.glob(
            str(target_poath.joinpath("neg/*.txt"))
        ))
        #posは1, negは0のラベル
        #(file_path, label)のtupleのリストを作成
        self.labeled_files = list(zip([0] * len(neg_files), neg_files)) + \
            list(zip([1]*len(pos_files), pos_files))
        @property
        def vocab_size(self):
            return len(self.vocab_array)
        
        def __len__(self):
            return len(self.labeled_files)

        def __getiem__(self, idx):
            label, f = self.labeled_files[idx]
            #ファイルのテキストデータを読み取って小文字に変換
            data = open(f).read().lower()
            #テキストデータをIDのリストに変換
            data = text2ids(data, self.vocab_dict)
            #IDのリストをTensorに変換
            data, n_tokens = list2tensor(data, self.max_len, self.padding)
            return data, label, n_tokens


class copas_reader():
    def __init__(path=""):
        self.path = path
        




import string
all_chars = string.printable
vocab_size = len(all_chars)

txt = ""
with open("./data/tinyshakespeare.txt", "r") as f:
    txt = f.read()

vocab_dict = 

vocab_dict = dict((c,i) for (i,c) in enumerate(all_chars))




#文字列を数値のリストに変換する関数
def str2ints(s, vocab_dict):
    return [vocab_dict[c] for c in s]

#数値のリストを文字列に変換する関数
def ints2str(x, vocab_array):
    return "".join([vocab_array[i] for i in x])
    


class CDataset(Dataset):
    def __init__(self, path , chunk_size=200):
        #ファイルを読み込み、数値のリストに変換する
        data = str2ints(open(path).read().strip(),vocab_dict)

        #Tensorに変換し、splitする
        data  = torch.tensor(data, dtype=torch.int64).split(chunk_size)

        #最後のchunkの長さをチェックして足りない場合には捨てる

        if len(data[-1]) < chunk_size:
            data = data[:-1]
        
        self.data = data
        self.n_chunks = len(self.data)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        return self.data[idx]

    

            

        

