import io
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#ターゲット
query = 'スパイダーマン'

#.text読み込み
fp = open('reviews.txt', encoding='utf-8')
titles = list()
reviews = list()
for line in fp:
    title, review = eval(line)
    titles.append(title)
    reviews.append(review)
fp.close()
np.set_printoptions(precision=5)#有効数字５桁

##対象の映画を(movie_id)へ##
movie_id = -1
for i in range(len(titles)):
    if titles[i] == query:
        movie_id = i
        break
if movie_id == -1:
    print('みつかりません')
    exit(0)

##特徴抽出##
vectorizer = TfidfVectorizer(max_df=0.03)#3％以上の文章に出てくる単語を除外
X = vectorizer.fit_transform(reviews).toarray()
features = vectorizer.get_feature_names()

list_sim=[]
list_titles=[]
num = 0
for i in range(len(titles)):
    idx = np.argsort(- X[i])
    X[i][idx[:num]] = 0.0

##類似度計算
def cos_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) *  np.linalg.norm(v2))#norm（ベクトルの大きさ)
for i in range(len(titles)):
    #sim = np.dot(X[movie_id], X[i])
    sim = cos_sim(X[movie_id], X[i])

    if sim > 0.0:
        list_titles.append(titles[i])
        #print('{:.8f}\t"{:s}"'.format(sim, titles[i]))
        list_sim.append(sim*(-1))
zipped=zip(list_sim,list_titles)
sorted_list = sorted(zipped, key = lambda t: t[0])

for i in range(11):
    print(sorted_list[i])


