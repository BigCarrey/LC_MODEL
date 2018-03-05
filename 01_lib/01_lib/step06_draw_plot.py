# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:42:46 2017

@author: Hank Kuang
@title: 统计图形
"""
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  

'''
drawPie        绘制饼形图
drawBar        绘制条形图
drawHistogram  绘制频数分布直方图
'''
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)  

  
def drawPie(s, labels=None, dropna=True):
    """
    Pie Plot for s
    -------------------------------------
    Params
    s: pandas Series
    lalels:labels of each unique value in s
    dropna:bool obj
    -------------------------------------
    Return
    show the plt object
    """
    counts = s.value_counts(dropna=dropna)
    if labels is None:
        labels = counts.index
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=labels, autopct='%1.2f%%', shadow=True, startangle=90, fontproperties=font)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    plt.show()
    

def drawBar(s, x_ticks=None, pct=False, dropna=False, horizontal=False):
    """
    bar plot for s
    -------------------------------------------
    Params
    s: pandas Series
    x_ticks: list, ticks in X axis
    pct: bool, True means trans data to odds
    dropna: bool obj,True means drop nan
    horizontal: bool, True means draw horizontal plot
    -------------------------------------------
    Return
    show the plt object
    """
    
    counts = s.value_counts(dropna=dropna)
    if pct == True:
        counts = counts/s.shape[0]
    ind = np.arange(counts.shape[0])
    if x_ticks is None:
        x_ticks = counts.index
    
    if horizontal == False:
        p = plt.bar(ind, counts)
        plt.ylabel('frequecy')
        plt.xticks(ind, tuple(counts.index), rotation=70, fontproperties=font)
    else:
        p = plt.barh(ind, counts, fontproperties=font)
        plt.xlabel('frequecy')
        plt.yticks(ind, tuple(counts.index), fontproperties=font)
    plt.title('Bar plot for %s' % s.name, fontproperties=font)
    
    plt.show()


def drawHistogram(s, num_bins=20, save=False, filename='myHist'):
    """
    plot histogram for s
    ---------------------------------------------
    Params
    s: pandas series
    num_bins: number of bins
    save: bool, is save? 
    filename png name
    ---------------------------------------------
    Return
    show the plt object
    """
    fig, ax = plt.subplots()
    mu = s.mean()
    sigma = s.std()
    # the histogram of the data
    n, bins, patches = ax.hist(s, num_bins, normed=1)
    
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, '--')
    ax.set_xlabel(s.name, fontproperties=font)
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of %s: $\mu=%.2f$, $\sigma=%.2f$' % (s.name, mu, sigma), fontproperties=font)
    
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    if save:
        plt.savefig(filename+'.png')
    plt.show()
    
    
    
def situation_ratio(data, var):
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)  

    for cm in var:
        fig = plt.figure()
        fig.set(alpha=0.2)  # 设定图表颜色alpha参数
        
        Customer_0 = df4[cm][df4.y == 0].value_counts()
        Customer_1 = df4[cm][df4.y == 1].value_counts()
        df=pd.DataFrame({u'坏客户':Customer_1, u'好客户':Customer_0})
        df.plot(kind='bar', stacked=True)
        plt.title(u"histogram of feature" +str(cm) , fontproperties=font)
        plt.xlabel(cm, fontproperties=font) 
        plt.ylabel(u"number", fontproperties=font) 
        plt.show()   
    
    
    
def Outliers(data):
    """
    plot Outliers for data
    ---------------------------------------------
    Params
    data: pandas series
    num_bins: number of bins
    save: bool, is save? 
    ---------------------------------------------
    Return
    show the plt object
    """

    plt.rcParams['font.sans-serif'] = ['SimHei']        #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False          #用来正常显示负号
    plt.figure(1, figsize=(6, 8))                     #可设定图像大小
    #plt.figure()                                         #建立图像
    #画箱线图，直接使用DataFrame的方法.代码到这为止,就已经可以显示带有异常值的箱型图了,
    #但为了标注出异常值的数值,还需要以下代码进行标注.
    p = data.boxplot() 
    plt.show()


  