# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:56:38 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#忽略弹出的warnings
import warnings
warnings.filterwarnings("ignore")

#自定义模块
import sys
sys.path.append(r"F:\TS\Lending_Club\01_lib\01_lib")
import step01_woe_iv
import step02_bining
import step03_statsmodels
import step04_moudle_evaluate
import step05_make_score
import step06_draw_plot

sys.path.append(r"F:\TS\Lending_Club\01_lib\02_lib")
import step01_feature_engine
import step02_modle_plot
import step03_built_modle

sys.path.append(r"F:\TS\Lending_Club\01_lib\03_lib")
import Step01_preprocess
import Step02_cal_iv
import Step03_model_helper
import Step04_model_plot

de1 = pd.read_csv(r'F:\TS\Lending_Club\02_data\LoanStats_data\LoanStats_2016Q4\LoanStats_2016Q4.csv')
de2 = pd.read_csv(r'F:\TS\Lending_Club\02_data\LoanStats_data\LoanStats_2017Q1\LoanStats_2017Q1.csv')
de3 = pd.read_csv(r'F:\TS\Lending_Club\02_data\LoanStats_data\LoanStats_2017Q2\LoanStats_2017Q2.csv')
df = pd.concat([de1, de2])

# =============================================================================
##统计每个月的放款个数
ls1 = pd.value_counts(df["loan_status"]).reset_index()


#########################
#定义好坏

#定义新函数 , 给出目标Y值
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)

    return colCoded

#把贷款状态LoanStatus编码为逾期=1, 正常=0:

pd.value_counts(df["loan_status"])
df["loan_status"] = coding(df["loan_status"], {'Current':0,'Fully Paid':0,
     'Late (31-120 days)':1,'Charged Off':1,
     'Late (16-30 days)':2,'In Grace Period':2,'Default':2})
print( '\nAfter Coding:')


df["loan_status"].groupby(df['issue_d']).value_counts()

###验证期间的样本
# =============================================================================
# de3["loan_status"] = coding(de3["loan_status"], {'Current':0,'Fully Paid':0,
#      'Late (31-120 days)':1,'Late (16-30 days)':1,'Charged Off':1,
#      'In Grace Period':2,'Default':2})
# print( '\nAfter Coding:')
# de3["loan_status"].groupby(de3['issue_d']).value_counts()
# =============================================================================

#删除无意义的字段
df1 = df.drop(['id','member_id','url','desc','zip_code','addr_state'], axis = 1)
df1 = df1.rename(columns = {'loan_status':'y'}, copy = False)
df2 = df1[(df1.y == 1)|(df1.y == 0)]
df2.groupby('y').size()
'''
y
0.0    186091
1.0     11618
6.24%
'''

##将y标签移到最后一列
last = df2['y']
df2.drop(labels=['y'], axis=1, inplace = True)
df2.insert(138, 'y', last)

#同值化检查，变量由139变为122个
df3,feature_primary_ratio = step01_feature_engine.select_primaryvalue_ratio(df2,ratiolimit = 0.942)

##针对缺失值进行IV值计算和分箱
null_data = df3[['mths_since_last_record','mths_since_recent_bc_dlq','mths_since_last_major_derog',
'mths_since_recent_revol_delinq','mths_since_last_delinq','il_util','mths_since_recent_inq','y']]

csvfile = r"F:\TS\Lending_Club\04_output\06_null_data\null_data.csv"
null_data.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')

#这些变量除了mths_since_recent_inq和il_util，其他IV值都是小于0.01的，可以直接删除


#查看缺失值情况，变量由122变为83个;其中几个40%-80%的缺失值其意义不大，无需进行missing编码
df4,null_ratio = step01_feature_engine.select_null_ratio(df3, ratiolimit = 0.40)



##看一下每个变量的描述性统计
ds = df.describe().T.reset_index()

##处理带有百分号的数据
df4['revol_util'] = df4['revol_util'].str.rstrip('%').astype('float')
df4['int_rate'] = df4['int_rate'].str.rstrip('%').astype('float')
df4['term'] = df4['term'].str.rstrip('months').astype('float')

##############################################
#对字符型数据进行编码和删除

##删掉一些无意义或者重复的变量，变量由83变为72个
##删除一些贷后的变量的，这些变量会向申请模型泄露信息

# =============================================================================
# next_pymnt_d : 客户下一个还款时间，没有意义
# emp_title ：数据分类太多，实用性不大
# last_pymnt_d ：最后一个还款日期，无意义
# last_credit_pull_d ：最近一个贷款的时间，没有意义
# sub_grade ： 与grade重复，分类太多
# title： title与purpose的信息基本重复，数据分类太多
# issue_d ： 放款时间，申请模型用不上
# earliest_cr_line : 贷款客户第一笔借款日期
# =============================================================================

'''
total_rec_prncp		  已还本金
total_rec_int		     已还利息
out_prncp	            剩余未偿本金总额
last_pymnt_d	        最后一个还款日
last_pymnt_amnt	     最后还款金额
next_pymnt_d	        下一个还款日
installment           每月分期金额
bc_open_to_buy        数据字典中未找到
percent_bc_gt_75      数据字典中未找到
tot_hi_cred_lim       无法译出真实意义
mths_since_recent_inq 数据字典中未找到
total_bc_limit        数据字典中未找到
'''

df5 = df4.drop(['emp_title','last_credit_pull_d','sub_grade','title',
                'issue_d','earliest_cr_line','funded_amnt_inv',
                'next_pymnt_d','last_pymnt_amnt','last_pymnt_d',
                'total_rec_prncp', 'out_prncp','out_prncp_inv',
                'total_pymnt','total_pymnt_inv','installment',
                'bc_open_to_buy','percent_bc_gt_75','tot_hi_cred_lim',
                'mths_since_recent_inq','total_bc_limit'], axis = 1)
    
#==============================================================================
#绘图
object_columns =df5.select_dtypes(include=["object"]) 
for i in object_columns:
    step06_draw_plot.drawBar(df5[i])
#==============================================================================


step01_feature_engine.watch_obj(df5)
mapping_dict = {"initial_list_status": 
                    {"w": 0,"f": 1,},
                "emp_length": 
                    {"10+ years": 11,"9 years": 10,"8 years": 9,
                     "7 years": 8,"6 years": 7,"5 years": 6,"4 years":5,
                     "3 years": 4,"2 years": 3,"1 year": 2,"< 1 year": 1,
                     "n/a": 0},
                "grade":
                    {"A": 0,"B": 1,"C": 2, "D": 3, "E": 4,"F": 5,"G": 6},
                "verification_status":
                    {"Not Verified":0,"Source Verified":1,"Verified":2},
                "purpose":
                    {"credit_card":0,"home_improvement":1,"debt_consolidation":2,       
                     "other":3,"major_purchase":4,"medical":5,"small_business":6,
                     "car":7,"vacation":8,"moving":9, "house":10, 
                     "renewable_energy":11,"wedding":12},
                "home_ownership":
                    {"MORTGAGE":0,"ANY":1,"NONE":2,"OWN":3,"RENT":4}} 
df6 = df5.replace(mapping_dict) 

step01_feature_engine.check_feature_binary(df6)  
    
#查看缺失值情况
#step01_feature_engine.fill_null_data(df3)

'''il_util缺失12%，但是缺失部分可以分类进入0，所以赋值为0'''

df6.isnull().sum(axis=0).sort_values(ascending=False)
null_ratio = step01_feature_engine.select_null_ratio(df6)

df7 = df6.fillna(0)
df7.isnull().sum(axis=0).sort_values(ascending=False)

#==============================================================================
 #绘图
var = list(df7.columns)
for i in var:
    step06_draw_plot.drawHistogram(df7[i])
#==============================================================================



last = df7['y']
df7.drop(labels=['y'], axis=1, inplace = True)
df7.insert(62, 'y', last)

#IV保留大于0.02的变量，63个变量保留26个
new_data,iv_value = step01_feature_engine.filter_iv(df7, group=10)

#对数据按照IV大小顺序进行排序，以便于使用fillter_pearson删除相关性较高里面IV值低的数据
list_value = iv_value[iv_value.ori_IV >= 0.02].var_name
iv_sort_columns = list(list_value.drop_duplicates())
df8 = new_data[iv_sort_columns]

iv_value.to_excel(r"F:\TS\Lending_Club\04_output\01_iv_value\iv_value_group01.xls")


csvfile = r"F:\TS\Lending_Club\05_middle\data_loan.csv"
df8.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')


df8 = pd.read_csv(r"F:\TS\Lending_Club\05_middle\data_loan.csv")

##皮尔森系数绘图，观察多重共线的变量
pearson_coef = step02_modle_plot.plot_pearson(df8)

#多变量分析，保留相关性低于阈值0.6的变量
#对产生的相关系数矩阵进行比较，并删除IV比较小的变量
per_col = step02_modle_plot.fillter_pearson(pearson_coef, threshold = 0.60)
print ('保留了变量有:',len(per_col))
print (per_col)   #26个变量,保留20个
df9 = df7[['grade','verification_status', 'acc_open_past_24mths',
       'inq_last_12m', 'inq_last_6mths','mths_since_rcnt_il', 
       'open_il_12m','il_util', 'mo_sin_rcnt_tl',
       'open_acc_6m', 'dti', 'all_util', 'inq_fi','mo_sin_rcnt_rev_tl_op',
       'tot_cur_bal', 'mort_acc', 'total_rev_hi_lim', 'y']]  


#df10 = df9.drop(['total_pymnt', 'total_pymnt_inv'], axis = 1)

pearson_coef = step02_modle_plot.plot_pearson(df9)  #再次观察共线情况


per_new_data,iv_new_value = step01_feature_engine.filter_iv(df9, group=10)
iv_new_value.to_excel(r"F:\TS\Lending_Club\04_output\01_iv_value\iv_value_group02.xls")


csvfile = r"F:\TS\Lending_Club\05_middle\data_loan.csv"
df9.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')


##利用卡方来进行最优分箱，但是需要花时间去手动调整分箱结果
#==============================================================================
#==============================================================================
#01基于卡方的最优分箱
#==============================================================================
# one_woe = pd.DataFrame([])
# new_col = list(df9.columns)
# new_col.remove('y')
#  
# for var in new_col:
#     new_woe = new_iv.ChiMerge(df9, var, 'y')
#     one_woe = one_woe.append(new_woe)
#     print(var)
# #     
# csvfile = r"F:\TS\Lending_Club\04_output\02_chimerge\chi1_iv_all01.csv"
# one_woe.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')
#==============================================================================


###随机森林查看特征的重要性
def rf_importance(data):
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    names = data.columns
    X, y = step01_feature_engine.x_y_data(data)

    clf=RandomForestClassifier(n_estimators=10,random_state=123)#构建分类随机森林分类器
    clf.fit(X, y) #对自变量和因变量进行拟合
    names, clf.feature_importances_
    for feature in zip(names, clf.feature_importances_):
        print(feature)

    #plt.style.use('fivethirtyeight')
    #plt.rcParams['figure.figsize'] = (10,5)
    
    ## feature importances 可视化##
    importances = clf.feature_importances_
    feat_names = names
    indices = np.argsort(importances)[::-1]
    #fig = plt.figure(figsize=(14,10))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=10)
    plt.xlim([-1, len(indices)])
    plt.show()


rf_importance(df9)

#==============================================================================
#==============================================================================
'''
按照woe中group回填的编码来进行训练模型
'''

#最终的数据,无多重共线以及IV值相对比较高的变量
data_loan = pd.read_csv(r"F:\TS\Lending_Club\05_middle\data_loan.csv",encoding = 'utf-8')

#经过等距分箱,卡方最优分箱,R语言中sanning包最优分箱,比较三者结果,进行手动分箱，最终IV>0.02有45个变量
loan_best_banning = data_loan[[
        "grade"
        ,"verification_status"
        ,"acc_open_past_24mths"
        ,"inq_last_12m"
        ,"inq_last_6mths"
       # ,"open_il_12m"
       #### ,"mths_since_rcnt_il"
        #,"mo_sin_rcnt_tl"
        ,"dti"
        ,"all_util"
        ,"il_util"
        ,"mo_sin_rcnt_rev_tl_op"
       # ,"inq_fi"
       ,"home_ownership"
       # ,"mort_acc"
       #### ,"tot_cur_bal"
       #### ,"total_rev_hi_lim"
        ,'y']]

#观察变量相关性
pearson_coef = step02_modle_plot.plot_pearson(loan_best_banning)

#导入WOE
woe = pd.read_excel(r"F:\TS\Lending_Club\05_middle\01_best_IV\02_best_IV_spi.xlsx")
print(len(woe.var_name.drop_duplicates()))


X, y = step01_feature_engine.x_y_data(loan_best_banning)

##将iv中分组的WOE值回填到 原始的样本中
  
new_col = list(X.columns)
bin_res_data=pd.DataFrame()
for var in new_col:
    bin_res = step03_built_modle.applyBinMap(X, woe, var)
    bin_res_data = pd.concat([bin_res_data,bin_res], axis = 1)
    

#未做one-hot编码
##构造X，y变量
#X, y = step01_feature_engine.x_y_data(bin_res_data)
X = bin_res_data
y = loan_best_banning['y']

#膨胀因子
vif_data = step01_feature_engine.judge_vif(X) 

##特征缩放

#==============================================================================
# Col = ["selfquery_cardquery_in3m",		
# "selfquery_cardquery_in3m",
# "selfquery_in3m_min_interval", #不具有单调性
# "housing_nature_g",
# "company_nature_g",
# "sum_carloan_line",
# "near_newopen_carloan",
# "far_open_loan",
# "desired_loan_amount",
# "education_g",
# "near_open_percosloan",
# "card_cardquery_rate",
# ]
# 
# from sklearn.preprocessing import MinMaxScaler
# 
# ms = MinMaxScaler()
# #区间缩放，返回值为缩放到[0, 1]区间的数据
# X[Col] = ms.fit_transform(X[Col])
# 
#==============================================================================


##处理样本不平衡；当样本过少的时候建议采用这个方法
X, y = step01_feature_engine.smote_data(X, y)

model = step03_built_modle.baseline_model(X, y)
'''
confusion_matrix 
 [[119931  66160]
 [  4040   7578]]
accuracy_score 0.644932704126
precision_score 0.102769264151
recall_score 0.652263728697
ROC_AUC is 0.704252105054
K-S score 0.296870197499
'''

#生成训练集测试集
X_train, X_test, y_train, y_test = step03_built_modle.train_test_split_data(X, y)
model = step03_built_modle.baseline_model(X_train, y_train)
'''
confusion_matrix 
 [[1005  581]
 [ 445 1148]]
accuracy_score 0.677256999056
precision_score 0.663967611336
recall_score 0.720652856246
ROC_AUC is 0.721041536546
K-S score 0.361780614906
'''

#网格搜索最优参数
#best_parameters = step03_built_modle.model_optimizing(X_train,y_train)

#利用最优参数建模
model = step03_built_modle.make_model(X_train,y_train,X_test, y_test,best_parameters=best_parameters)

'''
confusion_matrix 
 [[94621 53896]
 [50570 98658]]
accuracy_score 0.6491427228
precision_score 0.646708706425
recall_score 0.661122577532
ROC_AUC is 0.704619519737
K-S score 0.298737988924
'''
model = step03_built_modle.make_model(X_train,y_train,X_test, y_test,best_parameters=None)

##学习曲线


###截距和回归系数
formula = step03_built_modle.get_lr_formula(model, X)



##生成各变量的评分卡
scorecard = step03_built_modle.make_scorecard(formula, woe)

csvfile = r"F:\TS\Lending_Club\04_output\04_scorecard\scorecard_10_var_spi.csv"
scorecard.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')

scorecard = pd.read_csv(csvfile)


#生成编码对应的字典        
dict_code = step03_built_modle.change_dict_code(scorecard)


#==============================================================================
# ##再造一次X，y
# X, y = step01_feature_engine.x_y_data(loan_best_banning)
# 
# ##将iv中分组的WOE回填到 原始的样本中
#   
# new_col = list(X_last.columns)
# X_bin_data=pd.DataFrame()
# for var in new_col:
#     bin_res = step03_built_modle.applyBinMap(X_last, woe, var)
#     X_bin_data = pd.concat([X_bin_data,bin_res], axis = 1)
#==============================================================================


#生成分数
score_data = bin_res_data.replace(dict_code)
score_data["score_sum"] = score_data.sum(axis = 1)

#拼接y值
scorcarde_data = pd.concat([score_data, loan_best_banning['y']], axis =1)

iv_score_sum = step01_feature_engine.filter_iv(scorcarde_data, group=10)
score_group = iv_score_sum[1]
score_group.to_excel(r"F:\TS\Lending_Club\04_output\05_model_result\model_result10_spi.xlsx")


#画个图看下分数的分布情况
step06_draw_plot.drawHistogram(scorcarde_data['score_sum'])
v_feat = ['score_sum']
step02_modle_plot.prob_density(scorcarde_data, v_feat)

'''评分卡'''
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#plt.subplot2grid((2,3),(1,0), colspan=2)
scorcarde_data.score_sum[scorcarde_data.y == 0].plot(kind='kde')   
scorcarde_data.score_sum[scorcarde_data.y == 1].plot(kind='kde')
plt.xlabel(u"score_sum")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"Distribution of score_sum")
plt.legend((u'good', u'bad'),loc='best') # sets our legend for our graph.

#KS值>0.2就可认为模型有比较好的预测准确性











##建议先使用逐步法(P值限制为0.05，而不是0.01)
'''向前选择法(逐步法)'''
'''提高了选择最佳预测变量的能力，但是几乎不考虑一些低显著性的变量，
而且降低了处理速度，因为每一步都要考虑每一个变量的加入与删除'''
logit_instance, logit_model, logit_result, logit_result_0 = step03_statsmodels.logistic_reg(X, y, stepwise="FS")
desc, params, evaluate, quality = step03_statsmodels.logit_output(logit_instance, 
                                                                  logit_model, logit_result, logit_result_0)
print(desc,evaluate, quality,params)
"自变量 14个"
excelfile = r"F:\TS\Lending_Club\04_output\08_FS_params\FS_params.xlsx"
params_data = params.reset_index()
params_data.to_excel(excelfile)



'''向后淘汰法,不能使用过采样'''
'''从全部的备选变量中依次删除“最不显著”的变量，会保留一些低显著性的变量，
这些变量独立预测的能力不高，但是与其他变量结合会提升模型整体的预测能力'''
logit_instance, logit_model, logit_result, logit_result_0 = step03_statsmodels.logistic_reg(X, y, stepwise="BS")
desc, params, evaluate, quality = step03_statsmodels.logit_output(logit_instance,
                                                                  logit_model, logit_result, logit_result_0)
print(desc,evaluate, quality,params)
"自变量 14个"
excelfile = r"F:\TS\Lending_Club\04_output\07_BS_params\BS_params.xlsx"
params_data = params.reset_index()
params_data.to_excel(excelfile)



import statsmodels.api as sm
X_test = sm.add_constant(X[params.index.drop("const")])

step04_moudle_evaluate.plot_roc_curve(logit_result.predict(X_test),y)
ks_results, ks_ax=step04_moudle_evaluate.ks_stats(logit_result.predict(X_test), y, k=20)











#==============================================================================
#==============================================================================
##对训练集和测试机进行打分
##再造一次X，y；直接对数据进行分数编码
X_last, y_last = step01_feature_engine.x_y_data(loan_best_banning)
X_train, X_test, y_train, y_test = step03_built_modle.train_test_split_data(X_last, y_last)

scorecard = pd.read_csv(r"F:\TS\Lending_Club\04_output\04_scorecard\scorecard_9_var_spi.csv")

##将scoreacrd中分组的score回填到 原始的样本中
#针对X_train  
new_col = list(X_train.columns)
X_train_score=pd.DataFrame()
for var in new_col:
    bin_res = step03_built_modle.applymap_score(X_train, scorecard, var)
    X_train_score = pd.concat([X_train_score,bin_res], axis = 1)

X_train_score_data = X_train_score
#生成分数
X_train_score_data["score_sum"] = X_train_score_data.sum(axis = 1)

#拼接y值
scorcarde_data_X_train = pd.concat([X_train_score_data, y_train], axis =1)

iv_score_sum = step01_feature_engine.filter_iv(scorcarde_data_X_train, group=10)
score_group = iv_score_sum[1]
score_group.to_excel(r"F:\TS\Lending_Club\04_output\05_model_result\model_result14_.xlsx")


#画个图看下分数的分布情况
step06_draw_plot.drawHistogram(scorcarde_data['score_sum'])
v_feat = ['score_sum']
step02_modle_plot.prob_density(scorcarde_data, v_feat)

'''评分卡'''
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#plt.subplot2grid((2,3),(1,0), colspan=2)
scorcarde_data.score_sum[scorcarde_data.y == 0].plot(kind='kde')   
scorcarde_data.score_sum[scorcarde_data.y == 1].plot(kind='kde')
plt.xlabel(u"score_sum")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"Distribution of score_sum")
plt.legend((u'good', u'bad'),loc='best') # sets our legend for our graph.

#KS值>0.2就可认为模型有比较好的预测准确性



##将scoreacrd中分组的score回填到 原始的样本中
#针对X_test
new_col = list(X_test.columns)
X_score_data=pd.DataFrame()
for var in new_col:
    bin_res = step03_built_modle.applymap_score(X_last, scorecard, var)
    X_score_data = pd.concat([X_score_data,bin_res], axis = 1)


#生成分数
score_data = bin_res_data.replace(dict_code)
score_data["score_sum"] = score_data.sum(axis = 1)

#拼接y值
scorcarde_data = pd.concat([score_data, loan_best_banning['y']], axis =1)

iv_score_sum = step01_feature_engine.filter_iv(scorcarde_data, group=10)
score_group = iv_score_sum[1]
score_group.to_excel(r"F:\TS\Lending_Club\04_output\05_model_result\model_result14_.xlsx")


#==============================================================================
#==============================================================================
###对验证集进行打分

de3 = pd.read_csv(r'F:\TS\Lending_Club\02_data\LoanStats_data\LoanStats_2017Q2\LoanStats_2017Q2.csv')

# =============================================================================
##统计每个月的放款个数
ls3 = pd.value_counts(de3["loan_status"]).reset_index()

#定义新函数 , 给出目标Y值
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)

    return colCoded

#把贷款状态LoanStatus编码为逾期=1, 正常=0:

pd.value_counts(de3["loan_status"])
de3["loan_status"] = coding(de3["loan_status"], {'Current':0,'Fully Paid':0,
     'Late (31-120 days)':1,'Charged Off':1,
     'Late (16-30 days)':2,'In Grace Period':2,'Default':2})
print( '\nAfter Coding:')


de3["loan_status"].groupby(de3['issue_d']).value_counts()

###验证期间的样本

#删除无意义的字段
de4 = de3.rename(columns = {'loan_status':'y'}, copy = False)
de5 = de4[(de4.y == 1)|(de4.y == 0)]
de5.groupby('y').size()
'''
y
0.0    101018
1.0     3080
6.24%
'''


##处理带有百分号的数据
de5['revol_util'] = de5['revol_util'].str.rstrip('%').astype('float')
de5['int_rate'] = de5['int_rate'].str.rstrip('%').astype('float')
de5['term'] = de5['term'].str.rstrip('months').astype('float')



step01_feature_engine.watch_obj(de5)
mapping_dict = {"initial_list_status": 
                    {"w": 0,"f": 1,},
                "emp_length": 
                    {"10+ years": 11,"9 years": 10,"8 years": 9,
                     "7 years": 8,"6 years": 7,"5 years": 6,"4 years":5,
                     "3 years": 4,"2 years": 3,"1 year": 2,"< 1 year": 1,
                     "n/a": 0},
                "grade":
                    {"A": 0,"B": 1,"C": 2, "D": 3, "E": 4,"F": 5,"G": 6},
                "verification_status":
                    {"Not Verified":0,"Source Verified":1,"Verified":2},
                "purpose":
                    {"credit_card":0,"home_improvement":1,"debt_consolidation":2,       
                     "other":3,"major_purchase":4,"medical":5,"small_business":6,
                     "car":7,"vacation":8,"moving":9, "house":10, 
                     "renewable_energy":11,"wedding":12},
                "home_ownership":
                    {"MORTGAGE":0,"ANY":1,"NONE":2,"OWN":3,"RENT":4}} 
de6 = de5.replace(mapping_dict) 

step01_feature_engine.check_feature_binary(de6)  
    

de7 = de6.fillna(0)
de7.isnull().sum(axis=0).sort_values(ascending=False)


last = de7['y']
de7.drop(labels=['y'], axis=1, inplace = True)
de7.insert(144, 'y', last)

Verification = de7[[
        "grade"
        ,"verification_status"
        ,"acc_open_past_24mths"
        ,"inq_last_12m"
        ,"inq_last_6mths"
       # ,"open_il_12m"
       #### ,"mths_since_rcnt_il"
        #,"mo_sin_rcnt_tl"
        ,"dti"
        ,"all_util"
        ,"il_util"
        ,"mo_sin_rcnt_rev_tl_op"
       # ,"inq_fi"
        ,"home_ownership"
       # ,"mort_acc"
       #### ,"tot_cur_bal"
       #### ,"total_rev_hi_lim"
        ,'y']]

csvfile = r"F:\TS\Lending_Club\04_output\09_valid_data\valid_data.csv"
Verification.to_csv(csvfile,sep=',',index=False ,encoding = 'utf-8')

valid_data = pd.read_csv(r"F:\TS\Lending_Club\04_output\09_valid_data\valid_data.csv")

X_valid, y_valid = step01_feature_engine.x_y_data(valid_data)

##将scoreacrd中分组的score回填到 原始的样本中
#针对X_train  
new_col = list(X_valid.columns)
valid_score=pd.DataFrame()
for var in new_col:
    bin_res = step03_built_modle.applymap_score(X_valid, scorecard, var)
    valid_score = pd.concat([valid_score,bin_res], axis = 1)

#生成分数
valid_score["score_sum"] = valid_score.sum(axis = 1)

#拼接y值
valid_score_data = pd.concat([valid_score, valid_data['y']], axis =1)

iv_score_sum = step01_feature_engine.filter_iv(valid_score_data, group=10)
score_group = iv_score_sum[1]
score_group.to_excel(r"F:\TS\Lending_Club\04_output\09_valid_data\valid_model_result_spi_10.xlsx")


##计算SPI

dd1 = valid_score.drop(['score_sum'], axis = 1)
new_col = list(dd1.columns)
valid_spi=pd.DataFrame()
for var in new_col:
    lsvc = pd.value_counts(dd1[var]).reset_index()
    ls = lsvc.sort_values(by = ['index'], ascending=False)
    valid_spi = pd.concat([valid_spi,ls], axis = 1)
valid_spi.to_excel(r"F:\TS\Lending_Club\04_output\10_spi\spi_10.xlsx")
