# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:26:37 2017

@author: yichengfan
"""

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties  
plt.style.use('ggplot')  #风格设置接近R中的ggplot
import seaborn as sns    #数据可视化模块
sns.set_style('whitegrid')
import missingno as msno

#忽略弹出的warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score

import sys
sys.path.append(r"F:\TS\offline_model\lib")
import step01_feature_engine
import step02_modle_plot
import step03_built_modle


##绘出连续性变量的概率密度函数
def prob_density(data, v_feat):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)  
       
    plt.figure(figsize=(14,14*6))
    gs = gridspec.GridSpec(14, 1)
    for i, cn in enumerate(data[v_feat]):
        ax = plt.subplot(gs[i])
        sns.distplot(data[cn][data["y"] == 1], bins=50)
        sns.distplot(data[cn][data["y"] == 0], bins=100)
        ax.set_xlabel('')
        ax.set_title('histogram of feature: ' + str(cn), fontproperties=font)


###切分训练集和测试集    
def train_test_split(recut=False,random_state=0):
    """
    #model_selection.train_test_split随机分割train和test数据集
    #参数表示test_size=0.3 切分数据集70%训练，30%测试  
    """
    from sklearn.model_selection import train_test_split   
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=random_state)
    return X_train, X_test, y_train, y_test   #random_state是随机数的种子。但填0或不填，每次都会不一样。


###二分类数据正常和不正常数据之间的区别
def diff_two_classify(data):
    Xgood = data.loc[data["y"] == 1]
    Xbad = data.loc[data["y"] == 0]
      
    #good                
    correlationgood = Xgood.loc[:, data.columns != 'y'].corr()
    mask = np.zeros_like(correlationgood)
    indices = np.triu_indices_from(correlationgood)
    mask[indices] = True
    
    grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
    f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))
    
    cmap = sns.diverging_palette(220, 8, as_cmap=True)
    ax1 =sns.heatmap(correlationgood, ax = ax1, vmin = -1, vmax = 1, cmap = cmap, 
                     square = False, linewidths = 0.5, mask = mask, cbar = False)
    ax1.set_xticklabels(ax1.get_xticklabels(), size = 10); 
    ax1.set_yticklabels(ax1.get_yticklabels(), size = 10); 
    ax1.set_title('Good', size = 16)
    
    #bad
    correlationbad = Xbad.loc[:, data.columns != 'y'].corr()
    ax2 = sns.heatmap(correlationbad, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, 
                      square = False, linewidths = 0.5, mask = mask, yticklabels = False, 
                      cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', \
                                                   'ticks': [-1, -0.5, 0, 0.5, 1]})
    ax2.set_xticklabels(ax2.get_xticklabels(), size = 10); 
    ax2.set_title('Bad', size = 16);
    
    cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);

###绘出皮尔森相关系数图谱
def plot_pearson(data):
    '''
    我们在第一次降维的基础上，通过皮尔森相关性图谱找出冗余特征并将其剔除；
    同时，可以通过相关性图谱进一步引导我们选择特征的方向。
    小于0.4显著弱相关，0.4-0.75中等相关，大于0.75强相关
    '''
    colormap = plt.cm.viridis
    plt.figure(figsize=(18,18))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    pearson_coef = data.corr()
    sns.heatmap(pearson_coef,linewidths=0.1,vmax=1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True)
    return pearson_coef

def fillter_pearson(pearson_coef, threshold = 0.75):
    '''
    删除相关系数大于0.75或小于-0.75的变量
    但是列名必须先按照IV值排序
    '''
    b=-1
    for i in pearson_coef.index:
        rowdata = pearson_coef.ix[i,:]
        b = b + 1
        if any(np.abs(rowdata[:b]) >= threshold): #删除绝对值大于0.75的列
            pearson_coef = pearson_coef.drop(i)
            pearson_coef=pearson_coef.drop(i,axis=1)
            b = b - 1
    per_col = pearson_coef.index
    return per_col
    

###随机森林；特征的重要性排序
def rf_importance_plot(X,y):
    names = X.columns
    clf=RandomForestClassifier(n_estimators=10,random_state=123)#构建分类随机森林分类器
    clf.fit(X,y) #对自变量和因变量进行拟合
    feature_major = pd.DataFrame(clf.feature_importances_ , names)
    feature_major = feature_major.reset_index()
    feature_major = feature_major.rename(columns = {'index':'Col', 0:'value_retio'}, copy = False)  
    feature_major = feature_major.sort_values(by='value_retio',ascending=False)

    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (18,10)
    
    ## feature importances 可视化##
    importances = clf.feature_importances_
    feat_names = names
    indices = np.argsort(importances)[::-1]
    fig = plt.figure(figsize=(18,6))
    plt.title("Feature importances by RandomTreeClassifier")
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=12)
    plt.xlim([-1, len(indices)])
    plt.show()
    
    return feature_major


##绘制混淆矩阵,AUC曲线，KS曲线，以及评价模型

def __tool_sas_rank1(tmp_frame,group):
    '''
    这个按照 sas 公式实现rank分组功能，公式为
    floor(rank*k/(n+1))
    '''
    lenth = len(tmp_frame)
    tmp_frame['rank'] = tmp_frame.ix[:,1].rank(method='min')
    tmp_frame['group_num'] = tmp_frame.apply(lambda row : np.floor(row['rank']*group/(lenth+1)), axis=1)    

def model_evaluation_plot(model, X, y, ksgroup=20):        
    #plot Confusion Matrix 混淆矩阵
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    plt.figure(figsize=(10,6))
    plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    print("confusion_matrix \n",cm)
    print("accuracy_score %s"%(str(accuracy_score(y,y_pred))))
    print("precision_score %s"%(str(precision_score(y,y_pred))))
    print("recall_score %s"%(str(recall_score(y,y_pred))))


    """
    当预测效果较好时，ROC曲线凸向左上角的顶点。
    平移图中对角线，与ROC曲线相切，可以得到TPR较大而FPR较小的点。
    模型效果越好，则ROC曲线越远离对角线，极端的情形是ROC曲线经过（0，1）点，
    即将正例全部预测为正例而将负例全部预测为负例。
    ROC曲线下的面积可以定量地评价模型的效果，记作AUC，AUC越大则模型效果越好。
    """
    #plot ROC curve
    predictions_prob_forest = model.predict_proba(X)
    false_positive_rate,recall,threshold = roc_curve(y,predictions_prob_forest[:,1])
    roc_auc = auc(false_positive_rate,recall)
    plt.title('ROC curve')  
    plt.ylabel('True Positive rate')
    plt.xlabel('False Positive rate')        
    print("ROC_AUC is %s"%roc_auc)
    plt.plot(false_positive_rate,recall)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.show()

       
    #plot KS
    predictions_prob = pd.DataFrame(model.predict_proba(X))
    predictions_prob['y'] = y.get_values()
    __tool_sas_rank1(predictions_prob,ksgroup)                
    closPred1 = predictions_prob.groupby('group_num')[1].agg({'minPred1':min,'maxPred1':max})
    colsy = predictions_prob.groupby('group_num')['y'].agg({'bad':sum,'N':len})
    colsy['good'] = colsy['N']-colsy['bad']
    colscumy = colsy.cumsum(0) 
    colscumy = colscumy.rename(columns={'bad':'cum1','N': 'cumN','good':'cum0'}) 
    colscumy['cum1Percent'] = colscumy['cum1']/colscumy['cum1'].max()
    colscumy['cum0Percent'] = colscumy['cum0']/colscumy['cum0'].max() 
    colscumy['cumDiff'] = abs(colscumy['cum1Percent']-colscumy['cum0Percent'])
    ks_file = pd.concat([closPred1,colsy,colscumy],axis=1)
    ks_file['group'] = ks_file.index
    x = np.arange(1,ks_file.shape[0]+1)
    
    plt.plot(x,ks_file['cum0Percent'], label='cum0Percent',marker='o')
    plt.plot(x,ks_file['cum1Percent'], label='cum1Percent',marker='o')
    plt.plot(x,ks_file['cumDiff'], label='cumDiff',marker='o')
    plt.legend()
    plt.title('KS')
    plt.legend(loc='upper left')
    datadotxy=tuple(zip((x+0.2),ks_file['cumDiff']))
    for dotxy in datadotxy:
        plt.annotate(str(round(dotxy[1],2)),xy=dotxy)
        plt.xlabel(u"group", fontproperties='SimHei')
    #plt.savefig("F:\moudle\modle\\KS.png",dpi=2000)
    plt.show()

    p = pd.DataFrame(model.predict_proba(X),index = X.index)
    p['y'] = y
    proba_y0 = np.array(p[p['y']==0][1])
    proba_y1 = np.array(p[p['y']==1][1])
    ks = stats.ks_2samp(proba_y0,proba_y1)[0]   
    print("K-S score %s"%str(ks))
        
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=4, 
                        train_sizes=np.linspace(.05, 1., 16), verbose=0, plot=True):
    """
    用于判断模型是否过拟合，当模型在训练集上得分很高，但是在交叉验证集上得分很低时，模型过拟合
    画出data在某模型上的learning curve.如果两者得分有很大的差距，意味着可能过度拟合训练数据了
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    msyh = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\MSYHBD.TTF')           

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=None, n_jobs=4, train_sizes=np.linspace(.05, 1., 16), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(u"学习曲线", fontproperties=msyh)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数", fontproperties=msyh)
        plt.ylabel(u"得分", fontproperties=msyh)
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best", prop=msyh)

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff



"""
提升图和洛伦茨曲线
"""
def lift_lorenz(prob_y, y, k=10):
    """
    plot lift_lorenz curve 
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number 
    ----------------------------------
    lift_ax: lift chart
    lorenz_ax: lorenz curve
    """
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(prob_y, name='prob_y', index=y.index)], axis=1)
    datasets.columns = ["y", "prob_y"]
    datasets = datasets.sort_values(by="prob_y", axis=0, ascending=False)
    # 计算正案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    lift_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
          }
    lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['y'])
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    # 计算正例比例&累积正例比例
    lift_df['PerP'] = lift_df['Ptot']/P
    lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
    # 计算随机正例数、正例率以及累积随机正例率
    lift_df['randP'] = float(P)/k
    lift_df['PerRandP'] = lift_df['randP']/P
    lift_df.ix[0,:]=0
    lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
    lift_ax = lift_Chart(lift_df, k)
    lorenz_ax = lorenz_cruve(lift_df)
    return lift_ax, lorenz_ax


def lift_Chart(df, k):
    """
    middle function for lift_lorenz, plot lift Chart
    """
    #绘图变量
    PerP = df['PerP'][1:]
    PerRandP = df['PerRandP'][1:]
    #绘图参数
    fig, ax = plt.subplots()
    index = np.arange(k+1)[1:]
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, PerP, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Per_p')#正例比例
    rects2 = plt.bar(index + bar_width, PerRandP, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='random_P')#随机比例
    plt.xlabel('Group')
    plt.ylabel('Percent')
    plt.title('lift_Chart')
    plt.xticks(index + bar_width / 2, tuple(index))
    plt.legend()
    plt.tight_layout()
    plt.show()

def lorenz_cruve(df):
    """
    middle function for lift_lorenz, plot lorenz cruve
    """
    #准备绘图所需变量
    PerP_cum = df['PerP_cum']
    PerRandP_cum = df['PerRandP_cum']
    decilies = df['tile']
    #绘制洛伦茨曲线
    plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')#lorenz曲线
    plt.plot(decilies, PerRandP_cum, 'k-.', label='random')#随机
    plt.legend()
    plt.xlabel("decilis")#等份子集
    plt.title("lorenz_cruve", fontsize=10)#洛伦茨曲线
    plt.show()  