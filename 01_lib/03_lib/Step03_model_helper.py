# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:45:52 2017

@author: potato

#用于辅助建模，综合各种工具
"""
import sys
import pandas as pd
import numpy as np

sys.path.append(r"F:\moudle\lib")
import iv 
import model_evaluation_plot  

from sklearn.preprocessing import StandardScaler #标准化
from sklearn.model_selection import train_test_split   #数据集按比例切分为训练集和测试集
from sklearn.cross_validation import cross_val_score  #交叉验证 ; 对数据进行分类成train和test
from sklearn.pipeline import Pipeline  #参数集在新数据集（比如测试集）上的重复使用。管道机制实现了对全部步骤的流式化封装和管理
from sklearn.grid_search import GridSearchCV  #是自动调参，只要把参数输进去，就能给出最优化的结果和参数据
from sklearn.linear_model import RandomizedLogisticRegression   #随机逻辑回归,用于 稳定性选择
from sklearn.linear_model import RandomizedLasso    #随机lasso ，用于 稳定性选择
from sklearn.feature_selection import RFECV     ##特征选择(Feature Selection)  #递归特征消除Recursive feature elimination（RFE）
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier  #ensemble是集成方法
from sklearn.linear_model import LogisticRegression #逻辑回归学习速率(步长)参数，权重的正则化参数 
from sklearn.linear_model import SGDClassifier


class ModelHelper(object):
    
    def __refresh_Xy(self):
        """
        #所有函数以更新data为主，更新完data再更新X，y
        #data由数据和表现组成，表现列名为 'y'
        #根据data更新X，y的值。
        """
        columns = self.data.columns
        columns = columns[~(columns == "y")]
        self.X = self.data[columns]
        self.y = self.data['y']
        
    def __refresh_data(self):
        #根据X,y的值更新data
        self.data = pd.concat([pd.DataFrame(self.X),pd.DataFrame(self.y)],axis = 1)
    
    def __init__(self,data,model="LR"):
        self.data = data
        self.model_name = model
        if self.model_name == "LR":
            self.model = LogisticRegression()
        self.__refresh_Xy()
        # 默认按照 3/7分组
        #self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.3,random_state=0)
        
    def standard_scaler(self):
        """
        使用StandardScaler对X进行 标准化
        """
        #X标准化,SGDClassifier       
        scaler = StandardScaler()
        scaler.fit(self.X)
        self.X1 = pd.DataFrame(scaler.transform(self.X),index=self.X.index,columns = self.X.columns)
        self.__refresh_data()    
        
    def train_test_split(self,recut=False,random_state=0):
        """
        #model_selection.train_test_split分割train和test数据集
        #参数表示test_size=0.3 切分数据集70%训练，30%测试  
        """
        if recut == False:
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.3,random_state=random_state)
            return self.X_train, self.X_test, self.y_train, self.y_test   #random_state是随机数的种子。但填0或不填，每次都会不一样。
                
    def quick_make_model(self,model="LR",best_params=None):
        #快速建模方法
        '''#best_params 就是之前取得的最优化参数结果'''
        if model == "LR":
            if best_params == None:
                model = LogisticRegression()
            else:
                model = best_params['lr'] 
            model.fit(self.X_train, self.y_train)
            model_evaluation_plot.Model_Evaluation_Plot(model,self.X_test,self.y_test) #绘图，混淆矩阵，KS曲线，AUC
            
            return model
            
    @staticmethod
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

    
    def pick_variables(self,descover=True,method="rlr",threshold=0.25,auto_pick=True):#默认阈值0.25
        #挑选变量助手(特征选择)
        if method == "rlr":
            """
            #顶层特征选择算法
            #随机逻辑回归选择与y线性关系的变量(稳定性选择1)。
            #在不同数据子集和特征子集上运行特征选择算法(rlr)，最终汇总选择结果
            #不同的子集上建立模型，然后汇总最终确定特征得分
            稳定性选择是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。
            它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
            比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。
            理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。
            RandomizedLogisticRegression()
            fit(X, y)	Fit the model using X, y as training data.
            fit_transform(X[, y])	Fit to data, then transform it.
            get_params([deep])	Get parameters for this estimator.
            get_support([indices])	Get a mask, or integer index, of the features selected
            inverse_transform(X)	Reverse the transformation operation
            set_params(**params)	Set the parameters of this estimator.
            transform(X)	Reduce X to the selected features.
            """
            rlr = RandomizedLogisticRegression(selection_threshold=threshold)  #随机逻辑回归
            rlr.fit(self.X_train,self.y_train)
            scoretable = pd.DataFrame(rlr.all_scores_,index = self.X_train.columns,columns = ['var_score']) #汇总最终确定特征得分
            columns_need = list(self.X_train.columns[rlr.get_support()])  #	Get a mask, or integer index, of the features selected
            self.X_train = self.X_train[columns_need]
            self.X_test = self.X_test[columns_need]
            columns_need.append("y")
            if auto_pick:
                self.picked_data = self.data[columns_need]            
            return scoretable

    def pick_variables_bylist(self,columns_need):#默认阈值0.25
        #挑选变量助手
        self.picked_data = self.data[columns_need]   
   
    
    def model_optimizing(self,model="LR"):
        """
        使用LR
        """
        if model == "LR":
            pipline = Pipeline([('lr',LogisticRegression())
                    #('sgd',SGDClassifier(loss='log'))#LR
                    #('sgd',SGDClassifier(loss='hinge'))#SVM
                    #('svm',SVC()) 
                    ])
            parameters = {
              'lr__penalty': ('l1','l2'),'lr__C': (0.001,0.01,0.1,1.0,10,100),'lr__max_iter':(10,30,50,80,100,120,150,180),
              #'lr__multi_class': ('ovr','multinomial'),'lr__solver': ('newton-cg','lbfgs','liblinear','sag')
              #lr__penalty 正则化选择系数,#lr__C正则化的系数,lr__max_iter 收敛的最大次数,
              #lr__multi_class分类方式选择参数
#              #随机梯度下降分类器。alpha正则化的系数,n_iter在训练集训练的次数，learning_rate为什么是alpha的倒数
#              'sgd__alpha':(0.00001,0.000001,0.0001),'sgd__penalty':('l1','l2','elasticnet'),'sgd__n_iter':(10,50,5),  
#              #核函数，将数据映射到高维空间中，寻找可区分数据的高维空间的超平面
#              'svm__C':(2.5,1),'svm__kernel':('linear','poly','rbf'),
              }
            """
            网格搜索为自动化调参的常见技术之一，grid_search包提供了自动化调参的工具，包括GridSearchCV类
            #GridSearchCV 是自动调参，只要把参数输进去，就能给出最优化的结果和参数数据
            #sklearn模块的GridSearchCV模块，能够在指定的范围内自动搜索具有不同超参数的不同模型组合，有效解放注意力。
            """
            grid_search = GridSearchCV(pipline,parameters,n_jobs=6,scoring='recall',cv=3)
            """
            cv :交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
            n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
            """
            grid_search.fit(self.X_train, self.y_train) 
            print('Best score: %0.3f' % grid_search.best_score_)
            print('Best parameters set:')
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]))    
            return best_parameters


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X.shape
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    
    
    
    
    
    
    
    
    