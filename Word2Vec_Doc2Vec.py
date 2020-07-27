# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 20:36:48 2020

@author: 23909
"""
"""
进行文本无监督聚类操作
1.语料加载
2.分词
3.去停用词
4.抽取词向量特征
5.实战TF-IDF的中文文本K-means聚类
6.实战word2Vec的中文文本K-means聚类
7.结果可视化
"""
from gensim.models import Word2Vec
import jieba
#from application import *
from pre_data import *
#记载语料
datapath = ['E:\\MyMySql\\TextMining-master\\data\\coments1.csv', 'E:\\MyMySql\\TextMining-master\\data\\coments2.csv', 'E:\\MyMySql\\TextMining-master\\data\\coments3.csv', 'E:\\MyMySql\\TextMining-master\\data\\coments4.csv']
data_all = [ReadData(data_path) for data_path in datapath]
data_all = pd.concat(data_all).reset_index(drop=True)

#进行分词，去标点符号，以及停用词
sentences = CutWords(data_all,stopwords_path)

#对模型进行训练
model = Word2Vec(sentences)
model.save("model")#保存模型
model = Word2Vec.load("model")#加载模型

#Doc2Vec接收一个由LabeledSentence对象组成的迭代器作为其构造函数的输入参数。
#其中，LabeledSentence是Gensim内建的一个类，它接收两个List作为其初始化的参数:
#word list 和 label list













