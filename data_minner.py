# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:16:29 2020

@author: 23909
"""
import pandas as pd
def k_means(train_tfidf,n_clusters=10):
        """
        Parameters:
                train_tfidf:train set;
                n_clusters:the number of clusters;
                n_init:Number of time the k-means algorithm will be run with different centroid seeds;
                max_iter:Maximum number of iterations of the k-means algorithm for a single sun
                n_jobs:The number fo OpenMP threads to use for the computation.
                algorithm:{"suto","full",elkan},default="auto"
            
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.feature_extraction.text import TfidfVectorizer
          
        kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(train_tfidf)
        #lebels = kmeans.labels_
        results = kmeans.predict(train_tfidf)
        centers = kmeans.cluster_centers_
        score = ("%.2f"%kmeans.score(train_tfidf)).replace("-"," ")
        #score:Opposite of the value of the train set on the K-means objective 
        return results,centers,score

def k_means2(train_text,n_clusters=10):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    pipeline = Pipeline([("feature_extraction",TfidfVectorizer(max_df=0.4)),
                     ("clusterer",KMeans(n_clusters=n_clusters))
                     ])
    pipeline.fit(train_text)
    labels = pipeline.predict(train_text)
    #center = pipeline.center
    score = pipeline.score(train_text)
        
    from collections import Counter
    c = Counter(labels)
    for j,i in c.items(): 
        print("Cluster{} contains {} samples".format(j,i))
        
    return labels,score,pipeline


def print_topic(labels,n_clusters,pipeline):
    from collections import Counter
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    terms = pipeline.named_steps["feature_extraction"].get_feature_names()
    c = Counter(labels)
    #遍历所有的簇
    for cluster_number in range(n_clusters):
        print("Cluster {} contains {} samples".format(cluster_number, 
              c[cluster_number]))
        centroid = pipeline.named_steps["clusterer"].cluster_centers_[cluster_number]
        most_important = centroid.argsort()
        res_dcit = {}
        for i in range(5):
            term_index = most_important[-(i+1)]
            #print("{0},{1},score: {2:.4f}".format(str(i+1),str(terms[term_ index]),centroid[term_index]))
            #res_dict[str(i+1)] = [terms[term_ index],centroid[term_index]]
            #print(i+1,terms[term_ index],centroid[term_index])
    pass

def baiyes(text,labels):
    #1.抽取特征
    from sklearn.base import TransformerMixin
    from nltk import word_tokenize
    class NLTKBOW(TransformerMixin):
        def fit(self,text,labels=None):
            return self
        def transform(self,text):
            return [{word:True for word in word_tokenize(documnet)}
                    for documnet in text]
    
    #2.将字典转换为矩阵
    from sklearn.feature_extraction import DictVectorizer
    
    #3.训练朴素贝叶斯分类器
    from sklearn.naive_bayes import BernoulliNB
    
    #4.组装起来
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(
        [("bag-of-words",NLTKBOW()),
         ("vectorizer",DictVectorizer()),
         ("navie-bayes",BernoulliNB())]
        )
    #根据实际需要对target进行区间划分
    def f(x):
        if x in [1,2,3,4]:
            x = 0
        else:
            x = 1
        return x
    labels = pd.Series(labels)
    labels = labels.apply(f)
    #6.用F1评估
    #评分方式(f1或其他)、交叉验证(cv)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(pipeline,text.values,labels.values,scoring="f1",)
    import numpy as np
    print("Score:{:.3f}".format(np.mean(scores)))
    return scores




def simple_logistic_classify(normalized_X,y_tr,normalized_X_test,y_test,description):
    """
    辅助函数，用来训练逻辑回归分类器，并在测试数据上进行评分
    """
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression().fit(normalized_X,y_tr)
    s = m.score(normalized_X_test,y_test)
    print("Test score woth",description,"feature:",s)
    return m,s

#确定一个搜索网格，然后对每种特征集合执行5-折网格搜索
param_grid_ = {
    "C":[1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]
}


def Tunning_bow(X_tr,y_tr):
    """
    为一元词表示法进行分类器调优
    """
    from sklearn.linear_model import LogisticRegression
    import sklearn.model_selection as modsel
    bow_search = modsel.GridSearchCV(LogisticRegression(),cv=5,
                                    param_grid=param_grid_)
    
    bow_search.fit(X_tr,y_tr)
    bow = bow_search.cv_results_
    bow_max = bow.get("mean_test_score").max()#取最大值
    return bow_max

def Tunning_tfidf(X_tr_tfidf,y_tr):
    """
    为tf-idf表示法进行分类器调优
    """
    from sklearn.linear_model import LogisticRegression
    import sklearn.model_selection as modsel
    #tf-idf进行分类器调优
    tfidf_search = modsel.GridSearchCV(LogisticRegression(),cv=5,
                                      param_grid=param_grid_)
    tfidf_search.fit(X_tr_tfidf,y_tr)
    
    
    tfidf = tfidf_search.cv_results_
    tfidf_max = tfidf.get("mean_test_score").max()#取最大值
    return tfidf_max

def show_results(bow_search,tfidf_search):
    search_results = pd.DataFrame.from_dict({
    "bow":bow_search.cv_results_["mean_test_score"],
    "tfidf":tfidf_search.cv_results_["mean_test_score"],
    })

    #seaborn用来美化图形
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    sns.set_style("whitegrid")
    ax = sns.boxplot(data=search_results,width=0.4)
    ax.set_ylabel("Accuracy",size=14)
    ax.tick_params(labelsize=20)
    
    return plt.show()























