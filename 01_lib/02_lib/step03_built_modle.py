# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:52:19 2017

@author: Yichengfan
"""

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties  
plt.style.use('ggplot')  #风格设置接近R中的ggplot
import seaborn as sns    #数据可视化模块
sns.set_style('whitegrid')
import missingno as msno

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RandomizedLogisticRegression   #随机逻辑回归,用于 稳定性选择
from sklearn.pipeline import Pipeline  #参数集在新数据集（比如测试集）上的重复使用。管道机制实现了对全部步骤的流式化封装和管理
from sklearn.grid_search import GridSearchCV  
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split   
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(r"F:\TS\offline_model\lib")
import step01_feature_engine
import step02_modle_plot
import step03_built_modle


#忽略弹出的warnings
import warnings
warnings.filterwarnings("ignore")

def baseline_model(X, y):
    #快速建模方法
    '''#best_params 就是之前取得的最优化参数结果'''
    model = LogisticRegression(class_weight = 'balanced')
    model.fit(X, y)
    #model.get_support()
    y_pred = model.predict(X)
    print("Test set accuracy score: {:.5f}".format(accuracy_score(y, y_pred,)))
    print(classification_report(y, y_pred))
    print("The confusion_matrix is:\n", confusion_matrix(y, y_pred, labels=[0, 1]))
    
    step02_modle_plot.model_evaluation_plot(model, X, y)
    #step02_modle_plot.plot_learning_curve(model, X, y)
    return model
    
def train_test_split_data(X, y, random_state=0):
    """
    #model_selection.train_test_split分割train和test数据集
    #参数表示test_size=0.2 切分数据集80%训练，20%测试  
    """
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=random_state)
    return X_train, X_test, y_train, y_test   #random_state是随机数的种子。但填0或不填，每次都会不一样。


def model_optimizing(X_train,y_train):
    parameters = {'C': [0.01,0.1, 1, 10, 100, 1000,],
                            'penalty': [ 'l1', 'l2'],
      #'lr__max_iter':(10,30,50,80,100,120,150,180),
                  }
    """
    网格搜索为自动化调参的常见技术之一，grid_search包提供了自动化调参的工具，包括GridSearchCV类
    #GridSearchCV 是自动调参，只要把参数输进去，就能给出最优化的结果和参数数据
    #sklearn模块的GridSearchCV模块，能够在指定的范围内自动搜索具有不同超参数的不同模型组合，有效解放注意力。
    """
    grid_search = GridSearchCV(LogisticRegression(),parameters,n_jobs=4,scoring='recall',cv=10)
    """
    cv :交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
    n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
    """
    grid_search.fit(X_train, y_train) 
    print('Best score: %0.3f' % grid_search.best_score_)
    print('Best parameters set:')
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))    
    return best_parameters


def make_model(X_train,y_train,X_test, y_test,best_parameters=None):
    #best_params 就是之前取得的最优化参数结果
    if best_parameters == None:
        model = LogisticRegression()
    else:
        model = LogisticRegression(C = best_parameters['C'],
                                   penalty = best_parameters['penalty']
                                   #max_iter =  best_parameters['lr__max_iter'],
                                   )           
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test set accuracy score: {:.5f}".format(accuracy_score(y_test, y_pred,)))
    print(classification_report(y_test, y_pred))   
    print("The confusion_matrix is:\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))
    
    step02_modle_plot.model_evaluation_plot(model, X_train, y_train)
    return model


def get_lr_formula(model,X):
    '''返回回归系数和截距'''
    intercept = pd.DataFrame(model.intercept_) #截距
    coef = model.coef_.T   #模型(回归)系数(相关系数)
    coef = pd.DataFrame(coef)   
    formula = pd.concat([intercept,coef])
    index = ['Intercept']
    index = index + list(X.columns)
    formula.index = index
    formula.reset_index(inplace=True)
    formula.columns = [u'参数',u'估计值']
    return formula


def make_scorecard(formular,woe,basescore=600.0,base_odds=50.0/1.0,pdo=50.0):
    """
    一般行业规则，一般设定当odds为50时，score为600
    Odds翻倍时，score+20
    score = -(woe*b +a/n)*factor + offset/n
    factor = pdo/np.log(2)
    offset = basescore - factor*np.log(base_odds)

    """
#    #step6 生成评分卡
#    basescore = float(600)
#    base_odds = 50.0/1.0
#    pdo = float(50)
     #计算所需要的参数
     

    a = formular[formular[u"参数"] == "Intercept"].ix[0,u"估计值"]
    formular = formular.iloc[1:,:]
    n = float(len(formular))
    factor = pdo/np.log(2)
    offset = basescore - factor*np.log(base_odds)
    #生成评分卡
    scorecard = pd.DataFrame()
    for i in formular[u"参数"]:
        woe_frame = woe[woe['var_name'] == i][['var_name','interval','min','max','PctRec','bad_rate','WOE']]
        beta_i = formular[formular[u"参数"] == i][u"估计值"].iloc[0]
        #woe_frame['score'] = woe_frame['WOE'].apply(lambda woe : offset/n - factor*(a/n-np.abs(beta_i)*woe))
        woe_frame['score'] = woe_frame['WOE'].apply(lambda woe : offset/n - factor*(a/n+beta_i*woe))
        scorecard = pd.concat((scorecard,woe_frame),axis=0)
        
    return scorecard



def applyBinMap(X_data, bin_map, var):
    """
    将最优分箱的结果WOE值对原始数据进行编码
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    x = X_data[var]
    bin_map = bin_map[bin_map['var_name'] == var]
    bin_res = np.array([0] * x.shape[-1], dtype=float)
    
    for i in bin_map.index:
        upper = bin_map['max'][i]
        lower = bin_map['min'][i]
        if lower == upper:
            x1 = x[np.where(x >= lower)[0]]
        else:
            x1 = x[np.where((x >= lower) & (x < upper))[0]]  #会去筛选矩阵里面符合条件的值
        mask = np.in1d(x, x1)    #用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
        bin_res[mask] = bin_map['WOE'][i]   #将Ture的数据替换掉
    
    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name
    
    return bin_res


def change_dict_code(scorecard):
    '''
    构造group和score对应的一个字典,类似下面
    {score:{2: -1.17, 1: 20.04,0: 75.46}, }
    '''
    dict_code={}
    for i in scorecard.var_name.drop_duplicates():
        temp=scorecard[scorecard["var_name"]==i].set_index("WOE").T.to_dict("records")
        dict_code[i]=temp[5]
    return dict_code


def applymap_score(X_data, bin_map, var):
    """
    将评分卡结果对原始数据进行分数填充
    """
    x = X_data[var]
    bin_map = bin_map[bin_map['var_name'] == var]
    bin_res = np.array([0] * x.shape[-1], dtype=float)
    
    for i in bin_map.index:
        upper = bin_map['max'][i]
        lower = bin_map['min'][i]
        if lower == upper:
            x1 = x[np.where(x >= lower)[0]]
        else:
            x1 = x[np.where((x >= lower) & (x < upper))[0]]  #会去筛选矩阵里面符合条件的值
        mask = np.in1d(x, x1)    #用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
        bin_res[mask] = bin_map['score'][i]   #将Ture的数据替换掉
    
    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name
    
    return bin_res




