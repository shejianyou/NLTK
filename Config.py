# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 15:44:00 2020

@author: shejianyou
"""
import text_process as tp

stopwords_path = r"E:\MyMySql\chinese_dictionary-master\stopwords-master\hit_stopwords.txt"
simi_path = r"E:\MyMySql\chinese_dictionary-master\dict_synonym.txt"
english_dict = r""
baidu_stopwords = r"E:\MyMySql\chinese_dictionary-master\stopwords-master\baidu_stopwords.txt"

data_1_path = r"E:\MyMySql\TextMining-master\data\coments1.csv"
data_2_path = r"E:\MyMySql\TextMining-master\data\coments2.csv"
data_3_path = r"E:\MyMySql\TextMining-master\data\coments3.csv"
data_4_path = r"E:\MyMySql\TextMining-master\data\coments4.csv"

CNN_News = r"C:\Users\23909\Desktop\CNN-China.xlsx"

# 读取情感词典和待处理文件

# 读取情感词典，以列表的形式返回
#to create a file objects
with open(r"C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\emotion_dict\pos_dict.txt",encoding="utf-8") as f_pos:
    posdict = f_pos.readlines()#to iterate on the file objects

with open(r"C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\emotion_dict\neg_dict.txt",encoding="utf-8") as f_neg:
    negdict = f_neg.readlines()

# 读取程度副词词典，以列表的形式返回
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\most.txt',encoding="utf-8") as f_most:
    mostdict = f_most.readlines()#权值为2
#mostdict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\most.txt')   # 权值为2
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\very.txt',encoding="utf-8") as f_very:
    verydict = f_very.readlines()#权值为1.5
#verydict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\very.txt')   # 权值为1.5
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\more.txt',encoding="utf-8") as f_more:
    moredict = f_more.readlines()#权值为1.25
#moredict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\more.txt')   # 权值为1.25
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\ish.txt',encoding="utf-8") as f_ish:
    ishdict = f_ish.readlines()#权值为0.5
#ishdict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\ish.txt')   # 权值为0.5
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\insufficiently.txt',encoding="utf-8") as f_insuf:
    insufficientdict = f_insuf.readlines()#权值为0.25
#insufficientdict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\insufficiently.txt')  # 权值为0.25
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\over.txt',encoding="utf-8") as f_over:
    overdict = f_over.readlines()#权值为0.1
#overdict = tp.read_lines(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\over.txt')  # 权值为0.1
with open(r'C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\notDic.txt',encoding="utf-8") as f_invers:
    inversedict = f_invers.readlines()#权值为-1
#inversedict = tp.read_lines(r'‪C:\Users\23909\Desktop\Taobao_phone_sentiment_HowNet\Sentiment_dict\degree_dict\notDic.txt')  # 权值为-1



























