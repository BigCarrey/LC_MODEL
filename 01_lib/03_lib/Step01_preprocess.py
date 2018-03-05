# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 16:11:28 2017

@author: potato

用于合并dataframe 类型的 X，y
输入X，y后，可以执行一系列的 数值化，填充空值等操作
sklearn中的preproccessing库来进行数据预处理
preproccessing库的StandardScaler类对数据进行标准化
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocess(object):

    def __refresh_Xy(self):
        #所有函数以更新data为主，更新完data再更新X，y
        #data由数据和表现组成，表现列名为 'y'
        #根据data更新X，y的值。
        columns = self.data.columns
        # "~" 删掉数据中的y列
        columns = columns[~(columns == "y")]
        self.X = self.data[columns]
        self.y = self.data['y']
        
    def __refresh_data(self):
        #根据X,y的值更新data
        self.data = pd.concat([pd.DataFrame(self.X),pd.DataFrame(self.y)])
        
    
    def __init__(self,X,y):
        #X、y的index均为已经设定好的key
        self.X = X
        if type(y) == pd.DataFrame: #只支持1维的y
        #函数to_numeric如果字符串中有数字就会把数字剥离出来
            self.y = pd.to_numeric(y.iloc[:,0], errors='coerce')
        else:
            self.y = y
        self.y.name = 'y'
        self.data = Preprocess.conbine_X_y(self.X,self.y)
        self.__refresh_Xy()    
        self.total_badrate = self.y.sum()/float(len(self.y))
        self.hit_badreate = self.data['y'].sum()/float(len(self.data['y']))
        

    #使用@staticmethod或@classmethod，就可以不需要实例化，直接类名.方法名()来调用
    @staticmethod
    def conbine_X_y(X,y):
        #合并X，y。要求两个dataframe 的 index是key
        X.index.names = ['key']
        X = X.reset_index().drop_duplicates(subset='key', keep='first').set_index('key') #重复保留第一个
        y.index.names = ['key']
        y = y.reset_index().drop_duplicates(subset='key', keep='first').set_index('key') #重复保留第一个
        y = pd.DataFrame(y,columns = ['y'])
        data=pd.merge(X,y,how='inner',left_index=True,right_index=True) # X,y 合并要求两边皆有值，用inner join

        return data
        
    def find_nan_infinite(self):
        #查看数据里面有没有坏数据
        x = self.data.apply(lambda x : np.any(np.isnan(x)))  #若有null值，返回True，否则返回false
        y = ~self.data.apply(lambda x : np.all(np.isfinite(x)))   #np.isfinite 会对数据进行判断，如果是有限数据返回True
        frame = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
        frame.columns = ['have_nan','have_infinite']
        return frame
        
    def undigital_filler(self,fill_num = 0):
        #对于数据中出现的非数字值进行指定值填充
        def tt(str1):
            try:
                a = float(str1)
                return a
            except Exception as e:
                return fill_num
        
        def t(s):
            try:
                snew = s.astype(float)
                return snew
            except Exception as e:
                snew = s.apply(tt)
                return snew
                
        self.data = self.data.apply(t,axis=0)
        
        
    def primaryvalue_ratio(self,discover=True,ratiolimit = 0.9):
        #按照命中率进行筛选      
        #首先计算每个变量的命中率,这个命中率是指 维度中占比最大的值的占比       
        self.recordcount = self.data.shape[0]
        x = []
        #循环每一个列，并取出出现频率最大的那个值;index[0]是取列名,iloc[0]是取列名对应的值
        for col in self.X.columns:
            primaryvalue = self.data[col].value_counts().index[0]
            ratio = float(self.data[col].value_counts().iloc[0])/self.recordcount
            x.append([ratio,primaryvalue])       
        self.feature_primaryvalue_ratio = pd.DataFrame(x,index = self.X.columns)
        self.feature_primaryvalue_ratio.columns = ['primaryvalue_ratio','primaryvalue']
        
        if discover:
            #这个参数用于查看命中率
            return self.feature_primaryvalue_ratio
        else:
            needcol = self.feature_primaryvalue_ratio[self.feature_primaryvalue_ratio['primaryvalue_ratio']<=ratiolimit]
            needcol = list(needcol.index)
            if len(needcol)>0:                
                needcol.append("y")
                self.selected_data = self.data[needcol]
                
    
    def rule_get(self):
        #查找有价值的规则
        pass
        
        
   
