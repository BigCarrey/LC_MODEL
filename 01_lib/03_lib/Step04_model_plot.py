# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:15:55 2017

@author: potato

用于绘制模型效果的图的工具

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score
from scipy import stats

def __tool_sas_rank1(tmp_frame,group):
    '''
    这个按照 sas 公式实现rank分组功能，公式为
    floor(rank*k/(n+1))
    '''
    lenth = len(tmp_frame)
    tmp_frame['rank'] = tmp_frame.ix[:,1].rank(method='min')
    tmp_frame['group_num'] = tmp_frame.apply(lambda row : np.floor(row['rank']*group/(lenth+1)), axis=1)    

def Model_Evaluation_Plot(model,X,y,plot="all",ksgroup=20):
    #plot接受参数如下面字典
    graph_show = {"cm":False,"roc":False,"ks":False}
    
    if plot == "all":
        for k,v in graph_show.items():
            graph_show[k] = True
    else:
        graph_show[plot] = True #将指定的图设置为展示
 
        
#plot Confusion Matrix 混淆矩阵
    if graph_show["cm"]:
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        plt.figure(figsize=(10,6))
        fig, ax = plt.subplots()
        plt.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

        plt.title('Confusion matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        print("accuracy_score %s"%(str(accuracy_score(y,y_pred))))
        print("precision_score %s"%(str(precision_score(y,y_pred))))
        print("recall_score %s"%(str(recall_score(y,y_pred))))

#==============================================================================
# """
# 当预测效果较好时，ROC曲线凸向左上角的顶点。
# 平移图中对角线，与ROC曲线相切，可以得到TPR较大而FPR较小的点。
# 模型效果越好，则ROC曲线越远离对角线，极端的情形是ROC曲线经过（0，1）点，即将正例全部预测为正例而将负例全部预测为负例。
# ROC曲线下的面积可以定量地评价模型的效果，记作AUC，AUC越大则模型效果越好。
# """
#==============================================================================

#plot ROC curve
    if graph_show["roc"]:        
        predictions_prob_forest = model.predict_proba(X)
        false_positive_rate,recall,threshold = roc_curve(y,predictions_prob_forest[:,1])
        roc_auc = auc(false_positive_rate,recall)
        plt.title('ROC curve')  
        plt.ylabel('True Positive rate')
        plt.xlabel('False Positive rate')        
        plt.plot(false_positive_rate,recall)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
        print("ROC_AUC is %s"%roc_auc)


           
#plot KS
    if graph_show["ks"]:
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
        

'''
__name__ 是当前模块名，当模块被直接运行时模块名为 __main__ 。
当模块被直接运行时，以下代码块将被运行，当模块是被导入时，代码块不被运行。
'''            
if __name__ == "__main__":
    pre_prob = pd.DataFrame(lr.predict_proba(X_test))
    pre_prob['y'] = y_test.reset_index().iloc[:,-1]
    ks_file = model_plot(pre_prob)
    