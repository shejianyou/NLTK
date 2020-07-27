# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:29:20 2020

@author: 佘建友
"""
import pandas as pd
from pandas import *
import json

def load_data(filename):
    
    f = open(r"C:\Users\23909\qianbidao1\text.txt",encoding="gbk")
    js = []
    for line in f.readlines():
        line = line.replace("\\"," ")
        js.append(line)
    f.close
    news_df = pd.DataFrame(js)
    #df = pd.read_csv("‪C:\Users\23909\qianbidao3\text.txt",encoding="utf-8")
    #扩展为df
    new_df = news_df[0].str.split(":",expand=True)
    new_df = new_df.drop(columns=[0],axis=1)
    
    #按字段清洗,目前只清洗title,two_s
    one_s = new_df[1].str.replace("' r n"," ")
    one_s = new_df[1].str.replace(" "," ")
    one_s = new_df[1].str.replace("["," ")
    one_s = one_s.str.replace("' r n"," ")
    one_s = one_s.str.replace("r n                '], 'tag'"," ")
    one_s = one_s.str.replace("], 'tag'"," ")
    one_s = one_s.drop_duplicates()
    title = one_s
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
    return news_df



def n_gram(news_df,way=1):
    """
    英文文本调用分词
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
            normalized_X = scaler.transform(X_tr_tfidf)
            normalized_X_test = scaler.transform(X_te_tfidf)

            return normalized_X,normalized_X_test
        
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



def k_means(train_text,test_text,n=2):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    try:
        kmeans = KMeans(n_clusters=n,random_state=0).fit(train_text)
        #lebels = kmeans.labels_
        test = kmeans.predict(test_text)
        center = kmeans.cluster_centers_
        score = kmeans.score(test_text)
        score = ("%.2f"%score)
    
    except:
        #实例一个归一化数据
        scaler = StandardScaler(with_mean=False)
        #实例一个聚类对象
        kmeans = KMeans(n_clusters=4)
        pipeline = make_pipeline(scaler,kmeans)
        #对样本进行训练
        model = pipeline.fit(train_text)
        #对样本进行测试
        labels = pipeline.predict(train_text)
    #对测试样本进行评分
        score = silhouette_score(scaler,model.labels_)
    
    return test,score


def fetch_token(API_KEY,SECRET_KEY):
    import sys
    import json
    import base64
    import time
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
    # skip https auth
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
    
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req, timeout=5)
        result_str = f.read()
    except URLError as err:
        print(err)
    result_str = result_str.decode()
    result = json.loads(result_str)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if not 'brain_all_scope' in result['scope'].split(' '):
            print ('please ensure has check the  ability')
            exit()
        return result['access_token']
    else:
        print ('please overwrite the correct API_KEY and SECRET_KEY')
        exit()
    return result
    
def make_request(token,func_url,text):
    url = func_url + "?charset=UTF-8&access_token=" + token
    pass


def Lexanalysis(API_KEY,SECRET_KEY,News_Series,**kwargs):
    """
    Parameters:
            **kwargs:setConnectionTimeoutInMillis建立连接的超时时间
                    setSocketTimeoutInMillis通过打开的连接传输数据的超时时间（单位：毫秒）
            News_Series:新闻文本序列
    """
    from aip import AipNlp
    #实力一个NLP对象
    client = AipNlp(APP_ID, API_KEY, SECRET_KEY,**kwargs)
    text = News_Series
    result = client.lexer(k)#返回json格式的结果
    pass














def Lexanalysis(text_Series):
    pass




























