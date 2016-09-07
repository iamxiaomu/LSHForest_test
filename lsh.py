#coding:utf-8
# 使用lsh来处理 前缀树 
from sklearn.feature_extraction.text import TfidfVectorizer 
import jieba.posseg as pseg
from sklearn.neighbors import LSHForest
import os

def a_sub_b(a,b):
    ret = []
    for el in a:
        if el not in b:
            ret.append(el)
    return ret
stop = [line.strip().decode('utf-8') for line in open('stopword.txt').readlines() ]


#读文件
raw_documents=[]
walk = os.walk(os.path.realpath("/Users/muhongfen/sougou"))
for root, dirs, files in walk:
    for name in files:
        f = open(os.path.join(root, name), 'r')
    raw = str(os.path.join(root, name))+" "
    raw += f.read()
    raw_documents.append(raw)

#TfidfVectorizer 训练tfidf矩阵
tfidf_vectorizer = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 2), use_idf=1, smooth_idf=1,sublinear_tf=1)
train_documents = []
temp_documents = []  #用作输出 没有空格的格式
for item_text in raw_documents:
    item_str = "" 
    temp_str = ""
    item=(pseg.cut(item_text)) 
    for i in list(item):
    	if i.word not in list(stop):  #去掉停用词
    		item_str+=i.word
    		temp_str+=i.word
    		item_str+=" "      #tfidf_vectorizer.fit_transform的输入需要空格分隔的单词
    temp_documents.append(temp_str)
    train_documents.append(item_str)
x_train = tfidf_vectorizer.fit_transform(train_documents)


test_data_1 = '本报讯 全球最大个人电脑制造商戴尔公司８日说，由于市场竞争激烈，以及定价策略不当，该公司今年第一季度盈利预计有所下降。'\
'消息发布之后，戴尔股价一度下跌近６％，创下一年来的新低。戴尔公司估计，其第一季度收入约为１４２亿美元，每股收益３３美分。此前公司预测当季收入为１４２亿至１４６亿美元，'\
'每股收益３６至３８美分，而分析师平均预测戴尔同期收入为１４５．２亿美元，每股收益３８美分。为抢夺失去的市场份额，戴尔公司一些产品打折力度很大。戴尔公司首席执行官凯文·罗林斯在一份声明中说，'\
'公司在售后服务和产品质量方面一直在投资，同时不断下调价格。戴尔公司将于５月１８日公布第一季度的财报。'
test_cut_raw_1 = ""
item_test=(pseg.cut(test_data_1))
for j in list(item_test):
	test_cut_raw_1+=j.word
	test_cut_raw_1+=" "
x_test = tfidf_vectorizer.transform([test_cut_raw_1])

lshf = LSHForest(random_state=42)  #LSHForest训练数据
lshf.fit(x_train.toarray())
distances, indices = lshf.kneighbors(x_test.toarray(), n_neighbors=5)
print(distances)
print(indices)
for i in indices[0]:
	print i
	print temp_documents[i]