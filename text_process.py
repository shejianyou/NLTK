# -*- coding: utf-8 -*-

import jieba
import jieba.posseg as pseg


def read_file(file_path):
    """
    Function:
        input
    Parameters:
        a file path 
    Returns:
        a file object
    """
    file=open(file_path,encoding='utf-8').readlines()
    return file


def segmentation(sentence):
    """
    Function:
        分词
    Parameters:
        sentence
    Returns:
        seg_result--->list;word
    """
    seg_list = jieba.cut(sentence)
    seg_result = []
    [seg_result.append(w) for w in seg_list]
    #print seg_result[:]
    return seg_result


def postagger(sentence):
    """
    Function:
        分词，词性标注，词和词性构成一个元组
    Parameters:
        sentence
    Returns:
        pos_list--->list;element:词和词性构成的一个元组
    """
    pos_data = pseg.cut(sentence)
    pos_list = []
    for w in pos_data:
        if w.flag in ['n','v','a','ad','d','vd','vn']:
            pos_list.append((w.word, w.flag))
    #print pos_list[:]
    return pos_list


def cut_sentence(words):
    """
    Function:
        句子切分
    Parameters:
            words--->list,DataFrame or Series
    Returns:
            sents--->sentences;切分词语后
    """
    start = 0
    i = 0
    token = 'meaningless'
    sents = []
    punt_list = '!?~。！？～…， ,'
    
    for word in words:
        
        if word not in punt_list:   # 如果不是标点符号
            i += 1
            token = list(words[start:i+2]).pop()#to removes and returns the last item in the list
            
        elif word in punt_list and token in punt_list:  # 处理省略号
            #print "word2", word
            i += 1
            token = list(words[start:i+2]).pop()
            #print "token:", token
        else:
            #print "word3", word
            sents.append(words[start:i+1])   # 断句
            start = i + 1
            i += 1
    if start < len(words):   # 处理最后的部分
        sents.append(words[start:])
    return sents


def read_lines(filename):
    """
    Function:
        
    Parameters:
        fielname--->str
    Returns:
        lines--->list
    """
    fp = open(filename,'r', encoding='utf-8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines


def del_stopwords(seg_sent):
    """
    Function:
        去除停用词
    Parameters:
        seg_sent--->
    Returns:
        
    """
    stopwords = read_lines("./Sentiment_dict/emotion_dict/stop_words.txt")  # 读取停用词表
    new_sent = []   # 去除停用词后的句子
    for word in seg_sent:
        if word in stopwords:
            continue
        else:
            new_sent.append(word)
    return new_sent

# 获取六种权值的词，根据要求返回list，这个函数是为了配合Django的views下的函数使用
def read_quanzhi(request):
    """
    Function:
        
    Parameters:
        
    Returns:
    
    """
    result_dict = []
    if request == "one":
        result_dict = read_lines("./Sentiment_dict/degree_dict/most.txt")
    elif request == "two":
        result_dict = read_lines("./Sentiment_dict/degree_dict/very.txt")
    elif request == "three":
        result_dict = read_lines("./Sentiment_dict/degree_dict/more.txt")
    elif request == "four":
        result_dict = read_lines("./Sentiment_dict/degree_dict/ish.txt")
    elif request == "five":
        result_dict = read_lines("./Sentiment_dict/degree_dict/insufficiently.txt")
    elif request == "six":
        result_dict = read_lines("./Sentiment_dict/degree_dict/notDic.txt")
    else:
        pass
    return result_dict

if __name__ == '__main__':
    pass