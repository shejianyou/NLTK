# -*- coding: utf-8 -*-

from textprocess import cut_sentence,postagger,segmentation,read_lines

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from Config import *



# 2.程度副词处理，根据程度副词的种类不同乘以不同的权值
def match(word, sentiment_value):
    """
    Parameters:
        
    Returns:
    
    """
    if word in mostdict:
        sentiment_value *= 2.0
    elif word in verydict:
        sentiment_value *= 1.5
    elif word in moredict:
        sentiment_value *= 1.25
    elif word in ishdict:
        sentiment_value *= 0.5
    elif word in insufficientdict:
        sentiment_value *= 0.25
    elif word in overdict:
        sentiment_value *= 0.1
    elif word in inversedict:
        sentiment_value *= -1
    return sentiment_value

# 3.情感得分的最后处理，防止出现负数
def transform_to_positive_num(poscount, negcount):
    neg_count = 0
    if poscount < 0 and negcount >= 0:
        neg_count += negcount - poscount
        pos_count = 0
    elif negcount < 0 and poscount >= 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount < 0:
        neg_count = -poscount
        pos_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
    return pos_count, neg_count

# 求单条评论的情感倾向总得分
def single_review_sentiment_score(text):
    cut_paragraph = text.split('****')  # 对文本进行分段
    paragraph_scores = []
    cut_words=[]
    
    for paragraph in cut_paragraph:
        sentence_scores = []
        cuted_review = cut_sentence(paragraph)  # 句子切分，单独对每个句子进行分析
    
        for sent in cuted_review:
            seg_sent = segmentation(sent)   # 分词
            seg_sent =[item for item in seg_sent ]
            # seg_sent = tp.del_stopwords(seg_sent)[:]
            cut_words += seg_sent
            print('去除停用词结果',seg_sent)
            i = 0    # 记录扫描到的词的位置

            all_poscount = 0    # 记录该分句中的积极情感得分
            all_negcount = 0    # 记录该分句中的消极情感得分

            for word in seg_sent:   # 逐词分析
                poscount=0
                negcount=0
                if word in posdict:  # 如果是积极情感词
                    if i==0 or i ==len(seg_sent)-1:
                        poscount=1.2    #首尾情感积极得分+1.2
                    else:
                        poscount = 1   # 其他位置积极得分+1

                    if i-3>0:   #滑移窗口的设置，滑移步长为3，即判定情感词前面三个词语的程度值
                        for w in seg_sent[i-3:i+2]:  #情感词的位置小于3时，将扫描情感词前面的所有词语（词语数小于3）
                            poscount = match(w, poscount)
                    else:
                        for w in seg_sent[0:i+2]:
                            poscount = match(w, poscount)

                elif word in negdict:  # 如果是消极情感词
                    if i == 0 or i == len(seg_sent) - 1:
                        negcount = 1.2    #首尾消极情感得分+1.2
                    else:
                        negcount = 1    # 其他位置积极得分+1
                    if i - 3 > 0:  # 滑移窗口的设置，滑移步长为3，即判定情感词前面三个词语的程度值
                        for w in seg_sent[i-3:i+2]:   #情感词的位置小于3时，将扫描情感词前面的所有词语（词语数小于3）
                            negcount = match(w, negcount)
                    else:
                        for w in seg_sent[0:i+2]:
                            negcount = match(w, negcount)

                # 如果是感叹号，表示已经到本句句尾
                elif (word == "！"or word == "!") and (i ==len(seg_sent)-1):
                    for w2 in seg_sent[::-1]:  # 倒序扫描感叹号前的情感词，发现后权值+2，然后退出循环
                        if w2 in posdict:
                            poscount = 2
                            break
                        elif w2 in negdict:
                            negcount = 2
                            break
                # print(word,poscount,negcount)
                i += 1
                all_negcount+=negcount
                all_poscount+=poscount
            poscount, negcount=transform_to_positive_num(all_poscount,all_negcount)
            negcount=negcount*(-1)
            sentence_score=poscount+negcount
            print(sentence_score)

            sentence_scores.append(sentence_score)
        paragraph_score=np.mean(sentence_scores)
        paragraph_scores.append(paragraph_score)
    text_score=np.mean(paragraph_scores)
    text_score=round(float(text_score),3)
    if text_score>1:
        text_score=1
    elif text_score<-1:
        text_score=-1
    else:
        text_score=text_score
    cut_words=' '.join(cut_words)
    return text_score,cut_words



def run_score(data):
    """
    Parameters:
        data--->DataFrame
        savepath--->str
    Returns:
        data--->DataFrame;to have to column "sentiment" and "cut_word"
    """
    data = data.drop_duplicates()
    contents = data.评论
    scores = []
    cut_words=[]
    for index, content in enumerate(contents):
        content = str(content).replace('\n', '****')  # 用****替换换行符
        print(content)
        score,cut_word = single_review_sentiment_score(content)
        print('情感得分：{}\n'.format(score))
        scores.append(score)
        cut_words.append(cut_word)
    data['cut_word'] = cut_words
    data['sentiment'] = scores
    return data






