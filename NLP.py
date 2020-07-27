# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:29:20 2020

@author: 佘建友
"""
import pandas as pd
from pandas import *
import json


filename = r"C:\Users\23909\qianbidao1\text.txt"


def load_data(filename):
    
    f = open(filename,encoding="gbk")
    js = []
    for line in f.readlines():
        line = line.replace("\\"," ")
        js.append(line)
    f.close
    
    #扩展为df数据结构
    news_df = pd.DataFrame(js)
    new_df = news_df[0].str.split(":",expand=True)
    new_df = new_df.drop(columns=[0],axis=1)
    
    #按字段清洗,目前只清洗title,body
    one_s = new_df[1].str.replace("' r n"," ")
    one_s = new_df[1].str.replace(" "," ")
    one_s = new_df[1].str.replace("["," ")
    one_s = one_s.str.replace("' r n"," ")
    one_s = one_s.str.replace("r n                '], 'tag'"," ")
    one_s = one_s.str.replace("], 'tag'"," ")
    one_s = one_s.drop_duplicates()
    title = one_s
    title = pd.Series(title,index=list(range(len(title))))
    title = title.dropna()
    #选取特定长度的文本
    t_df = pd.DataFrame(title.values,index=list(range(len(title.values))),columns=["title"])
    t_df["title_len"] = [len(x) for x in t_df["title"]]
    title = t_df[t_df["title_len"]>=40]
    
    two_s = new_df[2]
    two_s = two_s.str.replace("['圈内热点'], '"," ")
    two_s = two_s.str.replace("["," ")
    two_s = two_s.str.replace("]"," ")
    two_s = two_s.str.replace("'avatar'"," ")
    two_s = two_s.str.replace("' ,"," ")
    two_s = two_s.str.replace("'"," ")
    two_s = two_s.str.replace(","," ")
    two_s = two_s.drop_duplicates()
    five_s = new_df[5]
    five_s = five_s.str.replace(""""\n"""," ")
    five_s = five_s.str.replace("""']}"""," ")
    five_s = five_s.str.replace("""["""," ")
    five_s = five_s.drop_duplicates()
    five_s = five_s.str.replace("""]}"""," ")
    five_s = five_s.str.replace("'"," ")
    five_s = five_s.dropna()
    five_s = five_s.str.replace("}"," ")
    text = five_s
    
    return title,text

def file_to_df(filename,n=20000):
    """
    Parameters
    ----------
    filename : str
        DESCRIPTION.
    n:num
        the rows of data
    Returns
    -------
    dataframe
    """
    
    f = open(filename,encoding="utf-8")
    js = []
    for i in range(n):
        js.append(json.loads(f.readline()))
    f.close()
    news_df = pd.DataFrame(js)
    text = news_df["body"]
    text = text.str.replace("\\"," ")
    text = text.str.lower()
    return text

def file_to_df2(filename):
    text_list = []
    with open(filename) as inf:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            #line = line.replace("\\"," ")
            print(json.loads(line)["title"])
            text_list.append(json.loads(line)["body"])
    return text_list

def csv_df(filename):
    df = pd.read_csv(filename)
    labels = df.Score
    text = df.Text
    return labels,text

def f(x):
    if x in [1,2,3,4]:
        x = 0
    else:
        x = 1
    return x







def n_gram_english(text,way=1):
    """
    英文文本调用分词
    停用词过滤(没有进行停用词过滤)
    """
    from sklearn.feature_extraction.text import CountVectorizer
    if way in [1,2,3]: 
        if way==1:
            #一元词
            bow_converter = CountVectorizer(token_pattern="(?u)\\b\\w+\\b")
            
            bow_converter.fit(text)
            words = bow_converter.get_feature_names()#拟合转换器，查看词汇表大小
            return words
        if way==2:
            #二元词
            bigram_converter = CountVectorizer(ngram_range=(2,2),
                                              token_pattern="(?u)\\b\\w+\\b")
            bigram_converter.fit(text)
            bigrams = bigram_converter.get_feature_names()
            return bigrams
        if way==3:
            #三元词
            trigram_converter = CountVectorizer(ngram_range=(3,3),
                                                token_pattern="(?u)\\b\\w+\\b")
            trigram_converter.fit(text)
            trigrams = trigram_converter.get_feature_names()
            return trigrams
    else:
        print("the inputing num is between 1 and 3!!!")

def create_word_matrix(text,way=1):
    """
    中文文本调用分词；
    轻量级调用根据需求调用方法一二三；
    重量级调用方法四
    """
    if way in [1,2,3,4]:
        #用词袋表示点评文本
        if way == 1:
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            X_tr_bow = vectorizer.fit_transform(text)
            X_te_bow = vectorizer.transform(text)#划为训练集和测试集
            #word_llen = len(vectorizer.vocabulary_)
            #print(vectorizer.get_feature_names())
            word_matrix = X_tr_bow.toarray()
            
            return X_tr_bow,X_te_bow,word_matrix
        
        if way == 2:
            #使用词袋矩阵创建tf-idf
            from sklearn.feature_extraction.text import TfidfTransformer
            
            tfidf_trfm = TfidfTransformer(norm=None)
            X_tr_tfidf = tfidf_trfm.fit_transform(X_tr_bow)
            X_te_tfidf = tfidf_trfm.transform(X_te_bow)
            
            return X_tr_tfidf,X_te_tfidf
        
        if way == 3:
            #对tf-idf进行归一化
            from sklearn.feature_extraction.text import CountVectorizer
            vectorizer = CountVectorizer()
            X_tr_bow = vectorizer.fit_transform(text)
            X_te_bow = vectorizer.transform(text)#划为训练集和测试集
            word_llen = len(vectorizer.vocabulary_)
            from sklearn.feature_extraction.text import TfidfTransformer
            from sklearn.preprocessing import StandardScaler
            tfidf_trfm = TfidfTransformer(norm=None)
            X_tr_tfidf = tfidf_trfm.fit_transform(X_tr_bow)
            X_te_tfidf = tfidf_trfm.transform(X_te_bow)
            from sklearn.preprocessing import Normalizer
            scaler = Normalizer().fit(X_tr_tfidf)
            normalized_X_tr= scaler.transform(X_tr_tfidf)
            normalized_X_test = scaler.transform(X_te_tfidf)

            return normalized_X_tr,normalized_X_test
        
        if way == 4:
            import jieba
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_list = [" ".join(jieba.cut(s,cut_all=False)) for s in text]#生成初始词表
            stpwrdpath = r"E:\MyMySql\feature\data\stopword.txt"
            with open(stpwrdpath, 'rb') as fp:
                stopword = fp.read().decode('utf-8')  # 提用词提取
            stopwordlist = stopword.splitlines()#将提用词转换为列表
            tfidf=TfidfVectorizer(stop_words=stopwordlist)#创建tfidf
            stopwordlist = stopword.splitlines()
            tfidf=TfidfVectorizer(stop_words=stopwordlist)
            X_tf =tfidf.fit_transform(all_list).toarray()
            X_tr_tfidf = X_tf[:-1]
            X_te_tfidf = X_tf[-1]
            
            return X_tr_tfidf,X_te_tfidf



def k_means(train_tfidf,train_text,n_clusters=10,way=1):
    """
    Parameters:
        train_tfidf:train set;
        n_clusters:the number of clusters;
        n_init:Number of time the k-means algorithm will be run with different centroid seeds;
        max_iter:Maximum number of iterations of the k-means algorithm for a single sun
        n_jobs:The number fo OpenMP threads to use for the computation.
        algorithm:{"suto","full",elkan},default="auto"
    
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    if way == 1:
        kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(train_tfidf)
        #lebels = kmeans.labels_
        labels = kmeans.predict(train_tfidf)
        center = kmeans.cluster_centers_
        score = ("%.2f"%kmeans.score(train_tfidf)).replace("-"," ")
        #score:Opposite of the value of the train set on the K-means objective 
        
        
    else:
        pipeline = Pipeline([("feature_extraction",TfidfVectorizer(max_df=0.4)),
                     ("clusterer",KMeans(n_clusters=n_clusters))
                     ])
        pipeline.fit(train_text)
        labels = pipeline.predict(train_text)
        #center = pipeline.center
        score = pipeline.score(train_text)
        
    from collections import Counter
    c = Counter(labels)
    for j,i in c.items(): 
        print("Cluster{} contains {} samples".format(j,i))
        
    return labels,score



#1.抽取特征
from sklearn.base import TransformerMixin
from nltk import word_tokenize
class NLTKBOW(TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return [{word:True for word in word_tokenize(documnet)}
                for documnet in X]

#2.将字典转换为矩阵
from sklearn.feature_extraction import DictVectorizer

#3.训练朴素贝叶斯分类器
from sklearn.naive_bayes import BernoulliNB

#4.组装起来
from sklearn.pipeline import Pipeline
pipeline = Pipeline(
    [("bag-of-words",NLTKBOW()),
     ("vectorizer",DictVectorizer()),
     ("navie-bayes",BernoulliNB())]
    )
#根据实际需要对target进行区间划分
def f(x):
    if x in [1,2,3,4]:
        x = 0
    else:
        x = 1
    return x
labels = labels.apply(f)
#6.用F1评估
#评分方式(f1或其他)、交叉验证(cv)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline,text.values,labels.values,scoring="f1",)
import numpy as np
print("Score:{:.3f}".format(np.mean(scores)))


class NLP(object):
    
    def __init__(self,APP_ID,API_KEY,SECRET_KEY):
        """
        Parameters:
        """
        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
    
    def commentTag(self,News_Series,type_=4):
        """
        Parameters:
                type_:which category you want to match
                News_Series:新闻文本序列
        -----------------------------------------------
        Returns:
            a dataframe which has columns：
                                        log_id
                                        prop
                                        adj
                                        sentiment
                                        begin_pos
                                        end_pos
                                        abstract
        """
        options = {}
        options["type"] = type_
        def f(News_Series):
            for text in News_Series:
                res = self.client.commentTag(text,options)#返回json格式的结果
                print(res)
                result = yield{
                    "log_id":res["log_id"],
                    "prop":res["items"][0]["prop"],
                    "adj":res["items"][0]["adj"],
                    "sentiment":res["items"][0]["sentiment"],
                    "begin_pos":res["items"][0]["begin_pos"],
                    "end_pos":res["items"][0]["end_pos"],
                    "abstract":res["items"][0]["abstract"]
                    }
            return result
        result = f(News_Series)
        res_df = pd.DataFrame(result)
        return res_df
    
    def sentiment(self,text_Series):
        """
        Parameters:
        ------------------------------
        Returns:
        ------------------------------
        """
        result = self.client.sentimentClassify(text_Series)
        return result
    
    def keyword(self,title,content):
        """
        Parameters:
        -------------------------------
        Returns:
        -------------------------------
        """
        result = self.client.keyword(title,content)
        return result
    
    def topic(self,title,content):
        """
        Parameters:
        --------------------------------
        Returns:
        --------------------------------
        """
        result = self.client.topic(title,content)
        return result
    
    



ValueError = """
Target is multiclass but average='binary'. Please choose another average
setting, one of [None, 'micro', 'macro', 'weighted'].
"""
binary = "This is applicabel only if targets are binary"
#one-hot-encoded_multiclasses = "average=None,average=micro"




























