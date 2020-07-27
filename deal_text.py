# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:29:21 2020

@author: 佘建友
"""
贝叶斯算法的本质="""
将具有n个词的邮件文本看做n维空间的一个点；
若P(yi|X)P(yi)谁更大，则邮件就属于i类型，
其中X={n1,n2,n3,n4,...,nn}.

但基于实际的需求，我们可能对文本语义要进行颗粒度的深刻理解。
故不仅需根据实际需求创建一元词、二元词、三元词，更有可能需要做词语搭配提取。

先提出两个概念：
信号：指对文本语义有帮助的n元词；噪声：指无帮助的或者会削弱我们对文本语义理解的n元词。

n元词的分类和词语搭配的提取通常不要求对文本进行深入地理解，故需要过滤掉没有实际意义地单词
以及进行词干提取。

过滤掉没有实际意义的单词通常需要业务需求与自身经验的结合。

综上本次实现n元词地分类，以及用词干提取

假设我们感兴趣的是找出一个问题中的所有名词短语。
为找出这些短语，首先我们切分出所有带词性的单词，
然后检查这些标记的临近词，找出按词性组合的词组，这些词组又称为“块”。

"""
#from 贝叶斯 import *
import json
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from pandas import DataFrame
import pandas as pd


filename=r"C:\Users\23909\Desktop\file\data\CNN-China.txt"


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
    return news_df


def n_gram(news_df,way=1):
    if way in [1,2,3]: 
        if way==1:
            #一元词
            bow_converter = CountVectorizer(token_pattern="(?u)\\b\\w+\\b")
            
            bow_converter.fit(news_df["body"])
            words = bow_converter.get_feature_names()#拟合转换器，查看词汇表大小
            return words
        if way==2:
            #二元词
            bigram_converter = CountVectorizer(ngram_range=(2,2),
                                              token_pattern="(?u)\\b\\w+\\b")
            bigram_converter.fit(news_df["body"])
            bigrams = bigram_converter.get_feature_names()
            return bigrams
        if way==3:
        #三元词
            trigram_converter = CountVectorizer(ngram_range=(3,3),
                                                token_pattern="(?u)\\b\\w+\\b")
            trigram_converter.fit(news_df["body"])
            trigrams = trigram_converter.get_feature_names()
            return trigrams
    else:
        print("the inputing num is between 1 and 3!!!")

def text_spacy(news_df,**kawarg):
    pass

def clean_words(words_list):
    df = DataFrame(words_list)
    df.apply(lambda df:df.str.lower())
    df.apply(lambda df:df.str.replace("\n",""))#用正则表达式更好
    #df.apply(lambda df:df.str.replace("\u\u",""))用正则表达式更好
    return df
def check_grammer(words_df):
    import Levenshtein
    minDistance = 0.8
    distance = -1
    lastWord=""
    pass
    

def gram_note(news_df):
    pass


from nltk.tokenize import sent_tokenize,word_tokenize

EXAMPLE_TEXT = """
Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard.
"""
print(sent_tokenize(EXAMPLE_TEXT))


























    
    
    
    



































