# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:13:55 2020

@author: 佘建友
"""
import pandas as pd
from pandas import *
import json


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

    
    




