# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 22:37:57 2020

@author: shejianyou
"""
from gensim import corpora,models,similarities
import gensim
import numpy as np
import matplotlib.pyplot as plt
#from pre_data import *
#from Word2Vec_Doc2Vec import *
plt.rcParams["font.sans-serif"] = [u"SimHei"]
plt.rcParams["axes.unicode_minus"] = False


"""
1.文件加载
2.jieba分词
3.去除停用词
4.构建词袋模型
5.LDA模型训练
6.结果可视化
"""
def LDA_topic(sentences,num_topics=10,top_n=10):
    """
    Parameters:
        doc2bow expects an array of unicode tokens on input
        sentences:每个词作为一个特征
        num_topics:主题数
        top_n:每个主题频率最高的n个词
    """
    dictionary = gensim.corpora.Dictionary(sentences)#构建词袋模型
    #元素为dict:key为词id,value为词频
    corpus = [dictionary.doc2bow(sentence) for sentence in sentences]
    lda = gensim.models.ldamodel.LdaModel(corpus,id2word=dictionary,num_topics=num_topics)
    return lda,dictionary

def plot_topic(lda,dictionary,num_show_term=8,num_topics=10):
    """
    Parameters:
        dictionary:词袋
        num_show_term:每个主题下显示几个词
        num_topics:主题数
    """
    plt.figure(figsize=(16,12))
    for i,k in enumerate(range(num_topics)):
        ax = plt.subplot(2,5,i+1)
        item_dis_all = lda.get_topic_terms(topicid=k)#get the representation of single topic
        num_show_term = num_show_term
        item_dis = np.array(item_dis_all[:num_show_term])
        ax.plot(range(num_show_term),item_dis[:,1],"b*")
        item_word_id = item_dis[:,0].astype(np.int)
        word = [dictionary.id2token[i] for i in item_word_id]
        ax.set_ylabel(u"概率")
        for j in range(num_show_term):
            ax.text(j,item_dis[j,1],word[j],bbox=dict(facecolor="green",alpha=0.1))
    plt.suptitle(u"9个主题及其7个主要词的概率")
    plt.show()
        





















