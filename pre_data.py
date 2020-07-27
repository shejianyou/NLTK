# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 14:52:51 2020

@author: shejianyou
"""
import pandas as pd
from pandas import *
from numpy import *
import numpy as np
import jieba
import re
import jieba.posseg as pseg
from Config import *
from csv import *
import csv
import gensim
from gensim import corpora,models
from gensim.matutils import corpus2dense
from sklearn.decomposition import PCA

def ReadData(data_path):
    """
    Function:
            读取数据
    """
    return pd.read_csv(data_path,index_col=False)

def ConcatData(path1,path2,path3,path4):
    """
    Function:
            合并数据
    """
    data_1 = ReadData(path1)
    data_2 = ReadData(path2)
    data_3 = ReadData(path3)
    data_4 = ReadData(path4)
    data_all = pd.concat([data1,data2,data3,data_4]
                         ).reset_index(drop=True)#以整数作为index
    return data_all


def GetNum(x):
    """
    Function:
            价格1、星级、评论数量 字段提取对应的数字信息
    """
    if type(x) == str:
        x = x.replace(","," ")
        return eval(re.findall("\d+\.?\d*",x)[0])#The return value is the result of the evaluated expression
    else:
        return np.nan

def GetPrice(x):
    """
    Function:
            价格2字段对应的数字信息提取
            价格2包含了一些其他无关的数据
    """
    if type(x) == str:
        
        if "￥" in x:
            x = x.replace(","," ")
            return eval(re.findall('\d+\.?\d*', x)[0])
        else:
            return np.nan
    else:
        return np.nan
    
def NameId(x):
    """
    Function:
            将类型变量替换为数字变量
            手机类型名称--->手机类型代码
    """
    temp = list(x.名称.unique())
    temp_dict = dict(zip(temp,[i for i in range(len(temp))]))
    x.名称= x.名称.map(temp_dict)
    
    return x,temp_dict

def GetPrice_2(x):
    """
    适用于字段价格2字段
    """
    if type(x) == str:
        
        if "￥" in x:
            x = x.replace(","," ")
            return eval(re.findall("\d+\.?\d*", x)[0])
        else:
            return np.nan
    else:
        return np.nan

def DataGropBy(x):
    """
    返回每条数据分组后的均值
    """
    return x.groupby("名称").mean()

def describe(x):
    pass


def CutWords(data_all,stopwords_path):
    """
    切词并去掉停用词
    """
    data_all.dropna(inplace=True)
    
    #加载分词后的评论
    cut_words_list = []
    for words in data_all.评论:
        cut_words = list(jieba.cut(words.replace(" "," ")))#精确模式识别新词
        cut_words_list.append(cut_words)
    cut_words_df = pd.DataFrame(
        {"words":cut_words_list}
        )
    lines = cut_words_df.words.tolist()
    
    #加载停用词
    stopwords = pd.read_csv(
        stopwords_path,quoting=csv.QUOTE_NONE,
        error_bad_lines=False,encoding="utf-8",
        names=["stopword"])
    stopwords = stopwords.stopword.values#定义停用词
    
    #标点符号
    punctuation = [",","。","：","；",".","‘","“","？","/","-","+","&","（","）","(",")"]
    #开始分词
    sentences = []
    for line in lines:
        segs = line
        words = []
        try:
            for word in segs:
                if word not in punctuation:
                    words.append(word)
            segs = [v for v in words if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(),segs))#去左右空格
            segs = list(filter(lambda x:x not in stopwords,segs))#去掉停用词
            sentences.append(segs)
        except Exception:
            print(line)
            continue
    
    return sentences

def GetAllCommet(data):
    """得到所有评论数据"""
    comments = ""
    for comment in data.评论:
        if type(comment) == str:
            comments += comment
        else:
            comments += ""
    return comments

def LoadStopWord(stopwords_path):
    stopwords = pd.read_csv(
        stopwords_path,quoting=csv.QUOTE_NONE,
        error_bad_lines=False,encoding="utf-8"
        ,names=["stopword"])
    return stopwords.stopword.tolist()

def SimiWords(simi_path):
    """
    同义词替换制作成字典
    """
    simi_ = []
    simi_words_df = pd.read_csv(simi_path, names=["classes","labels","nums"],sep=" ")
    simi_words_df = simi_words_df.drop_duplicates("labels",keep="first",inplace=False)
    simi_words = simi_words_df.labels.tolist()#Return a list of the values
    simi_words_iterm = simi_words_df.groupby("classes").labels
    
    for _,syn in simi_words_iterm:#the groupby object attribute:a dict
        for _ in syn.tolist():#
            simi_.append(syn.tolist()[0])
            
    return dict(zip(simi_words,simi_)),simi_words

def CutWords_list(comment):
    """
    每一条数据评论切词.后续副词正反向词权重准备
    """
    if type(comment) == str:
        
        words_list = list(jieba.cut(comment))
        return words_list
    
    else:
        return []
    
def GetNoun(commen_data,stopwords_path,simi_path):
    """
    处理分词后的词语:停用词、名词过滤、相关词替换
    """
    count = 1
    noun_words = []
    stop_words = LoadStopWord(stopwords_path)
    dict_simi,simi_words = SimiWords(simi_path)
    word_list = []
    
    for commen in commen_data:
        count += 1
        if type(commen) == str:
            for word,flag in pseg.cut(commen):
                if (flag == "n") and (word not in stop_words) and (len(word)>=2):
                    if word in simi_words:
                        word_list.append(dict_simi.get(word))
                        #Return the value for key if the key is in the dict
                    else:
                        word_list.append(word)
            noun_words.append(word_list)
        else:
            noun_words.append([])
    return noun_words

def get_category(content_lines,sentences):
    """
    Parameters:
        content_lines: corpus
        sentences: an empty list
        
    -----------------------------------------------
    """
    import random
    for line in content_lines:
        category = random.randint(0, 3)
        try:
            sentences.append((" ".join(line),category))
        except:
            print(line)
            continue
    return sentences

def Tf_idf(sentences):
    dictionary = corpora.Dictionary(sentences)#创建词典
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]#创建文档词频向量
    tfidf_model = models.TfidfModel(corpus)#计算tf-idf值
    corpus_tfidf = tfidf_model[corpus]
    corpus_matrix = corpus2dense(corpus_tfidf, len(dictionary))#转换矩阵形式
    return corpus_matrix

def split_text_data(sentences,my_tuple=(1,1)):
    """
    sentences:
            the corpus
    my_tuple:
            the ngrams of size 1 and 2
    """
    #把语料分割为训练集和测试集
    from sklearn.model_selection import train_test_split
    x,y = zip(*sentences)
    x_train,x_test,y_train,y_test = train_test_split(x, y,random_state=1256)
    
    #抽取词袋模型特征
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(
        analyzer="word",#tokenise by character ngram
        ngram_range = my_tuple,
        max_features = 400,#keep the most common 4000 ngrams
        )
    
    #把训练数据转换为词袋模型
    vec.fit(x_train)
    x_train = vec.transform(x_train)
    vec.fit(x_test)
    x_test = vec.transform(x_test)
    #y_train = vec.fit(y_train)
    #y_train = vec.fit(y_train)
    return x_train,x_test,y_train,y_test
    
def PCA_(text_matrix,n_1=10):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca = PCA(n_1)#载入N维
    pca.fit(text_matrix)
    var = pca.explained_variance_
    newData = pca.transform(text_matrix)
    #newData = TSNE(n_2).fit_transform(newData)#再使用TSNE
    return newData,var

  
    

    
    
    
    








































