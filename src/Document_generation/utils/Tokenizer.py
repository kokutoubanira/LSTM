class japanese_Tokenizer(object):
    '''BERT用の文章の単語分割クラスを実装'''

    def __init__(self, vocab_file, do_lower_case=True):
        '''
        vocab_file：ボキャブラリーへのパス
        do_lower_case：前処理で単語を小文字化するかどうか
        '''

        # ボキャブラリーのロード
        self.vocab, self.ids_to_tokens = load_vocab_sent(vocab_file)

        # 分割処理の関数をフォルダ「utils」からimoprt、sub-wordで単語分割を行う
        never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]","[unk]", "[sep]", "[pad]", "[cls]", "[mask]")
        # (注釈)上記の単語は途中で分割させない。これで一つの単語とみなす

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

        self.SPT = SentencePieceTokenizer()


    def tokenize(self, text):
        '''文章を単語に分割する関数'''
        split_tokens = []  # 分割後の単語たち
        split_tokens = self.SPT.tokenize(text)
        split_tokens = self.convert_by_vocab(self.vocab, split_tokens)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """分割された単語リストをIDに変換する関数"""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """IDを単語に変換する関数"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def convert_by_vocab(self, vocab, items, unk_info="[UNK]"):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in items:
            if item in vocab:
                output.append(item)
            else:
                output.append(unk_info)
        return output
