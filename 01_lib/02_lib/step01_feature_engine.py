# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:29:10 2017

@author: yichengfan
"""

import numpy as np
import pandas as pd
from scipy.interpolate import lagrange  #导入拉格朗日函数

from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块

from sklearn.preprocessing import StandardScaler # 导入模块
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression   #随机逻辑回归,用于 稳定性选择

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')  #风格设置接近R中的ggplot
import seaborn as sns    #数据可视化模块
sns.set_style('whitegrid')
import missingno as msno

#忽略弹出的warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(r"F:\model\lib")
import iv
import model_evaluation_plot
import model_helper
import preprocess

##判断变量中是字符型的变量
def check_feature_binary(data):
    object_columns_df =data.select_dtypes(include=["object"]) #筛选数据类型为object的变量
    print(object_columns_df.iloc[0])
    
#观察各个离散值的分布情况
def watch_obj(data):
    objectColumns = data.select_dtypes(include=["object"]).columns
    var = data[objectColumns].columns
    for v in var:
        print('\nFrequency count for variable {0}'.format(v))
        print(data[v].value_counts())
    print(data[objectColumns].shape)

##对于缺失值的处理，一般来说先判定缺失的数据是否有意义。
##从确实信息可以发现，本次数据集缺失值较多的属性对我们模型预测意义不大
##统计每列属性缺失值的数量,删除缺失值过高的比例，如缺失比例达到0.3'''
def select_null_ratio(data, ratiolimit = 0.3):
    '''删除缺失值比例大于0.3的特征'''
    df_null = data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data))
    null_ratio = pd.DataFrame(df_null)
    null_ratio = null_ratio.reset_index() #重置索引
    null_ratio = null_ratio.rename(columns = {'index':'Col', 0:'value_retio'}, copy = False)    
    df_nullre = null_ratio[null_ratio.value_retio < ratiolimit] #删掉缺失值>0.3的数据   
    refesh_data = data[list(df_nullre['Col'])] 
    return refesh_data, null_ratio


###变量同值性观察;删除维度占比过大值过高的比例的函数
##当特征值都是离散型变量的时候这种方法才能用
def select_primaryvalue_ratio(data, ratiolimit = 0.95):
    '''按照命中率进行筛选,首先计算每个变量的命中率,这个命中率是指维度中占比最大的值的占比  '''     
    recordcount = data.shape[0]
    x = []
    #循环每一个列，并取出出现频率最大的那个值;index[0]是取列名,iloc[0]是取列名对应的值
    for col in data.columns:
        primaryvalue = data[col].value_counts().index[0]
        ratio = float(data[col].value_counts().iloc[0])/recordcount
        x.append([ratio,primaryvalue])       
    feature_primaryvalue_ratio = pd.DataFrame(x,index = data.columns)
    feature_primaryvalue_ratio.columns = ['primaryvalue_ratio','primaryvalue']
        
    needcol = feature_primaryvalue_ratio[feature_primaryvalue_ratio['primaryvalue_ratio']<ratiolimit]
    needcol = needcol.reset_index()
    select_data = data[list(needcol['index'])]
    return select_data, feature_primaryvalue_ratio  


##移除方差过小的的变量,抛弃了0个(参数threshold为方差的阈值),需要输入的数据无缺失值
##假设我们有一个有布尔特征的数据集，然后我们想去掉那些超过90%的样本都是0（或者1）的特征
def select_var(data):
    '''特征是否发散,方差选择法'''
    from sklearn.feature_selection import VarianceThreshold
    
    vt = VarianceThreshold(threshold = .9 * (1 - .9))
    x_vt=vt.fit_transform(data)
    vt.get_support()
    vt_columns = data.columns[vt.get_support()]
    data_new = data[vt_columns]
    return data_new


##卡方检验
def chi_square_test():
    '''
    chi2,卡方统计量，X中特征取值必须非负。
    卡方检验用来测度随机变量之间的依赖关系。
    通过卡方检验得到的特征之间是最可能独立的随机变量，因此这些特征的区分度很高。
    '''
    from sklearn.feature_selection import SelectKBest,chi2
    
    #选择K个最好的特征，返回选择特征后的数据
    SelectKBest(chi2, k=15).fit_transform(X, y)
    return X, y


##缺失值可视化和填充缺失值
def fill_null_data(data):
    #缺失值可视化
    data.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)
    numColumns = data.select_dtypes(include=[np.number]).columns
    msno.matrix(data[numColumns]) #缺失值可视化

    from sklearn.preprocessing import Imputer
    imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)  # 针对axis=0 列来处理,填充众数
    imr = imr.fit(data[numColumns])
    data[numColumns] = imr.transform(data[numColumns])
    return data


def str_ratio(data,var):
    #计算每个分类变量的逾期率情况，看情况合并

    str_value = pd.DataFrame([])
    for v in var:
        Customer_0 = data[v][data.y == 0].value_counts()
        Customer_1 = data[v][data.y == 1].value_counts()
        df_ratio=pd.DataFrame({u'坏客户':Customer_1, u'好客户':Customer_0})
        df_ratio['total'] = df_ratio['坏客户']+df_ratio['好客户']
        df_ratio['bad_ratio'] = df_ratio['坏客户']/(df_ratio['total'])
        df_ratio['pct_rec'] =  df_ratio['total']/data.shape[0]
        df_ratio['varname'] = v
        str_value = pd.concat([str_value,df_ratio], axis = 0)
    str_value = str_value.reset_index()
    return str_value


def abnormal_check(data):
##异常值检测
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    
    for var in data.columns:
    
        plt.figure() #建立图像
        data_var = data[[var]]
        data_var.plot(kind = 'box')
        #p = data_var.boxplot(return_type='dict') #画箱线图，直接使用DataFrame的方法
#==============================================================================
#         x = p['fliers'][0].get_xdata() # 'flies'即为异常值的标签
#         y = p['fliers'][0].get_ydata()
#         y.sort() #从小到大排序，该方法直接改变原对象
#     
#         #其中有些相近的点，注解会出现重叠，难以看清，需要一些技巧来控制。
#         #以下参数都是经过调试的，需要具体问题具体调试。
#         for i in range(len(x)): 
#           if i>0:
#             plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.05 -0.8/(y[i]-y[i-1]),y[i]))
#           else:
#             plt.annotate(y[i], xy = (x[i],y[i]), xytext=(x[i]+0.08,y[i]))
#==============================================================================
        
        plt.show() #展示箱线图
 

##离群点检测
def select_alone_data(data, k = 3, threshold = 2, iteration = 500):
    """
    k          ##聚类的类别
    threshold  ##离散点的阈值
    iteration  ##聚类最大循环次数
    """
    data_zs = 1.0 * (data - data.mean())/data.std() #数据标准化
    
    model = KMeans(n_clusters = k, n_jobs = 4, max_iter = iteration) #分为k类,并发数4
    model.fit(data_zs)   #开始聚类
    
    ##标准化数据及其类别
    r = pd.concat([data_zs, pd.Series(model.labels_, index = data.index)], axis = 1)
    ##对于每个样本对应的类别
    r.columns = list(data.columns) + [u'聚类类别']
    
    norm = []
    for i in range(k):
        norm_tmp = r[['R', 'F', 'M']][r[u'聚类类别'] == i] - model.cluster_centers_[i]
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis = 1)  #求出绝对值
        norm_tmp.append(norm/norm.median())  #求相对距离并添加
        
    norm = pd.concat(norm) #合并
    
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei'] #用来显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    #用来显示负号
    norm[norm <= threshold].plot(style = 'go')  #正常点
    
    discrete_points = norm[norm > threshold]  #离群点
    discrete_points.plot(style = 'ro') 
    
    for i in range(len(discrete_points)):  #离群点做标记
        id = discrete_points.index[i]
        n = discrete_points.iloc[i]
        plt.annotate('(%s, %0.2f)'%(id, n), xy = (id, n), xytext = (id,n))
        plt.xlabel(u'编号')
        plt.ylabel(u'相对距离')
        plt.show()
        
 
###拉格朗日插值法
def ployinterp_columns(s, n, k = 5):
    '''
    s为列向量, n为被插值的位置,k为取前后的数据个数,默认为5
    '''
    y = s[list(range(n - k, n)) + list(n+1, n+1+k)]  #取数
    y = y[y.notnull()]  #删除空值
    lag = lagrange(y.index, list(y))(n)  #插值并返回插值结果
    return lag

def larange_missing_value(data):   
    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]: #如果为空即插值
                data[i][j] = ployinterp_columns(data[i], j)
    return data


def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


##卡方阈值作为终止分箱条件
from sklearn.feature_selection import chi2
def ChiMerge_MinChisq(df, col, target, confidenceVal = 3.841):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param confidenceVal: the specified chi-square thresold, by default the degree of freedom is 1 and using confidence level as 0.95
    :return: the splitted bins
    '''
    colLevels = set(df[col])
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total':total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad':bad})
    regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
    regroup.reset_index(level=None, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B*1.0/N
    colLevels =sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]
    groupNum  = len(groupIntervals)
    while(1):   #the termination condition: all the attributes form a single interval; or all the chi-square is above the threshould
        if len(groupIntervals) == 1:
            break
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = chi2(df2, 'total','bad',overallRate)
            chisqList.append(chisq) 
        min_position = chisqList.index(min(chisqList))
        if min(chisqList) >=confidenceVal:
            break
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position -1
        else:
            if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    return groupIntervals
 

### 利用IV值来删除不重要的特征
def filter_iv(data, group=10):
    iv_value,all_iv_detail = iv.cal_iv(data, group=group)
    ##利用IV值，先删除掉IV值<0.02的特征
    '''IV值小于0.02,变量的预测能力太弱'''
    list_value = iv_value[iv_value.ori_IV <= 0.02].var_name
    filter_data = iv_value[['var_name','ori_IV']].drop_duplicates()
    print(filter_data)
    
    new_list = list(set(list_value))
    print('小于0.02的变量有:',len(new_list))
    print(new_list)
    #new_list.sort(key = list_value.index)   
    drop_list = new_list
    new_data = data.drop(drop_list, axis = 1)
    return new_data, iv_value

##逻辑回归对共线性敏感，需要判断VIF
def judge_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    vif_data = pd.DataFrame([])
    vif_data["VIF_Factor"] = [vif(X.values, i) for i in range(X.shape[1])]
    vif_data["features"] = X.columns
    return vif_data

def dummy_onehot(data,n_columns):
    dummy_df = pd.get_dummies(data[n_columns])# 用get_dummies进行one hot编码
    dummy_data = pd.concat([data, dummy_df], axis=1) 
    return dummy_data, dummy_df

def x_y_data(data):
    #构建X特征变量和Y目标变量
    x_feature = list(data.columns)
    x_feature.remove('y')
    X = data[x_feature]
    y = data['y']
    return X, y


###标准化
def standard_scaler(X):
    """
    使用StandardScaler对X进行 标准化
    """
    Col = X.columns
    sc =StandardScaler() # 初始化缩放器
    X[Col] =sc.fit_transform(X[Col])  #对数据进行标准化
    return X
 
 
###递归消除算法来暴力选择特征,顶层特征选择算法
def wrapper_data(X, y, n_features_to_select=15):
    '''
    递归特征消除的主要思想是反复的构建模型(如SVM或者回归模型)然后选出最好的(或者最差的)的特征(可以根据系数来选)，
    把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。
    这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。
    RFE的稳定性很大程度上取决于在迭代的时候底层用哪种模型。
    例如，假如RFE采用的普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；
    假如采用的是Ridge，而用Ridge正则化的回归是稳定的，那么RFE就是稳定的。    
   '''     
    ####wrapper递归消除算法
    model = LogisticRegression()  # 建立逻辑回归分类器
    # 建立递归特征消除筛选器,n_features_to_select是要选择的特征数,默认保留一一半的特征数
    rfe = RFE(model, n_features_to_select) #通过递归选择特征，选择15个特征
    rfe = rfe.fit(X, y)
    print(rfe.support_)
    print(rfe.ranking_) #ranking 为 1代表被选中，其他则未被代表未被选中a
    
    col_filter = X.columns[rfe.support_] #通过布尔值筛选首次降维后的变量
    print(col_filter) # 查看通过递归特征消除法筛选的变量
    X = X[col_filter]
    return X, y


###稳定性选择,顶层特征选择算法
def rdlg_variables(X, y, threshold=0.25):#默认阈值0.25
    """
    #随机逻辑回归选择与y线性关系的变量(稳定性选择1)。
    #在不同数据子集和特征子集上运行特征选择算法(rlr)，最终汇总选择结果
    #不同的子集上建立模型，然后汇总最终确定特征得分
    稳定性选择是一种基于二次抽样和选择算法相结合较新的方法，选择算法可以是回归、SVM或其他类似的方法。
    它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
    比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。
    理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。
    总的来说，好的特征不会因为有相似的特征、关联特征而得分为0，这跟Lasso是不同的。
    对于特征选择任务，在许多数据集和环境下，稳定性选择往往是性能最好的方法之一。
    """

    rlr = RandomizedLogisticRegression(selection_threshold = threshold)  #随机逻辑回归
    rlr.fit(X, y)
    scoretable = pd.DataFrame(rlr.all_scores_, index = X.columns) #汇总最终确定特征得分
    scoretable = scoretable.reset_index()    
    scoretable = scoretable.rename(columns = {'index':'Col', 0:'value_retio'}, copy = False)    
    df_score = scoretable[scoretable.value_retio > threshold] #删掉缺失值<0.25的数据   
    refesh_data = X[list(df_score['Col'])] 
         
    return scoretable,refesh_data


#过采样
def smote_data(X, y):
    Col = list(X.columns)
    n_sample = y.shape[0]
    n_pos_sample = y[y == 0].shape[0]
    n_neg_sample = y[y == 1].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample, n_pos_sample/n_sample, n_neg_sample/n_sample))
    print('特征维数：', X.shape[1])
    
    # 处理不平衡数据
    sm = SMOTE(random_state=42)    # 处理过采样的方法
    X, y = sm.fit_sample(X, y)
    X = pd.DataFrame(X,columns = Col)
    y = pd.Series(y)
    print('通过SMOTE方法平衡正负样本后')
    n_sample = y.shape[0]
    n_pos_sample = y[y == 0].shape[0]
    n_neg_sample = y[y == 1].shape[0]
    print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample, n_pos_sample/n_sample, n_neg_sample/n_sample))
    return X, y


