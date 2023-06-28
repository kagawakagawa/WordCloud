import pandas as pd
import MeCab
import re
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm
import numpy as np

import MeCab
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from collections import defaultdict

# MeCabオブジェクトの生成
mt = MeCab.Tagger('')
mt.parse('')

# トピック数の設定
NUM_TOPICS = 3

if __name__ == "__main__":
    # トレーニングデータの読み込み
    # train_texts は二次元のリスト
    # テキストデータを一件ずつ分かち書き（名詞、動詞、形容詞に限定）して train_texts に格納するだけ
    train_texts = []
    with open('./train.txt', 'r') as f:
        for line in f:
            text = []
            node = mt.parseToNode(line.strip())
            while node:
                fields = node.feature.split(",")
                if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                    text.append(node.surface)
                node = node.next
            train_texts.append(text)

    # モデル作成
    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]
    lda = LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

    # テストデータ読み込み
    # test_texts は train_texts と同じフォーマット
    test_texts = []
    raw_test_texts = []
    with open('./test.txt', 'r') as f:
        for line in f:
            text = []
            raw_test_texts.append(line.strip())
            node = mt.parseToNode(line.strip())
            while node:
                fields = node.feature.split(",")
                if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                    text.append(node.surface)
                node = node.next
            test_texts.append(text)

    # テストデータをモデルに掛ける
    score_by_topic = defaultdict(int)
    test_corpus = [dictionary.doc2bow(text) for text in test_texts]

    # クラスタリング結果を出力
    for unseen_doc, raw_train_text in zip(test_corpus, raw_test_texts):
        print(raw_train_text, end='\t')
        for topic, score in lda[unseen_doc]:
            score_by_topic[int(topic)] = float(score)
        for i in range(NUM_TOPICS):
            print('{:.2f}'.format(score_by_topic[i]), end='\t')
        print()
