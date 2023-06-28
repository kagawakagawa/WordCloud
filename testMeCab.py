from wordcloud import WordCloud
import os
# テキストファイル読み込み
f = open(os.path.sep.join(['corpora', 'en_abs_1.txt']), encoding='utf-8')
raw = f.read()
f.close()

wc = WordCloud().generate(raw)
wc.to_file("wc-1.png")

