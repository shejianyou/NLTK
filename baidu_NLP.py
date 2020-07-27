# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:17:11 2020

@author: 佘建友
"""
from aip import AipNlp
from pandas import *
import pandas as pd

APP_ID = "20156929"
API_KEY = "NZr5LGhMhU8EEFQqPCKCZUvZ"
SECRET_KEY = "LT4RCm08hYh4gHOC5ZrktgnL63KwGgot"


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
    
    def sentiment(self,text_series):
        import time
        """
        Parameters:
            text:string
        ------------------------------
        Returns:
            DataFrame which has columns:
                                text--->str;
                                sentiment--->int;表示情感极性分类结果,0:负向,1:中性,2:正向
                                confidence--->float;表示分类的置信度
                                positive_prob--->float;表示属于积极类别的概率
                                negative_prob--->float;表示属于消极类别的概率
        ------------------------------
        """
        
        df_sentiment = pd.DataFrame()
        
        results = []
        for text in text_series:
            results.append(self.client.sentimentClassify(text))#速度问题
            time.sleep(3)#防止请求过快
            
        text = [result["text"] for result in results]
        sentiment = [result["items"][0]["sentiment"] for result in results]
        confidence = [result["items"][0]["confidence"] for result in results]
        positive_prob = [result["items"][0]["positive_prob"] for result in results]
        negative_prob = [result["items"][0]["negative_prob"] for result in results]
            
        df_sentiment["text"] = text
        df_sentiment["sentiment"] = sentiment
        df_sentiment["confidence"] = confidence
        df_sentiment["positive_prob"] = positive_prob
        df_sentiment["negative_prob"] = negative_prob
        
        return df_sentiment,text,sentiment
    
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
    


