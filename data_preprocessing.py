# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 20:41:49 2020

@author: 佘建友
"""
import pandas as pd
from pandas import *

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
    使用词袋
        方法一：
                无过滤
        方法二：
                英文---基于频率的过滤
        方法三：
                中文---基于停用词进行过滤
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


