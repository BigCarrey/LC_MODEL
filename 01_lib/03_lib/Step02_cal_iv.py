# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 17:13:10 2017

@author: potato
"""
import pandas as pd
import numpy as np

def tool_group_rank(tmp_frame, k = 10):
    '''
    http://stackoverflow.com/questions/23976176/ranks-within-groupby-in-pandas
    这个是去重的值，排序rank后，等距分组。
#==============================================================================
#     '''
#     from sklearn.cluster import KMeans
#     kmodel = KMeans(n_clusters = k, n_jobs = 4) #n_jobs是并行数，一般等于CPU数较好
#     
#     for j in tmp_frame.columns:
#         
#         kmodel.fit(tmp_frame[j].reshape((len(tmp_frame[j]), 1)))
#         tt = pd.DataFrame(kmodel.cluster_centers_).sort_values(0) #输出聚类中心,并排序
#         ww = pd.rolling_mean(tt, 2).iloc[1:]  #相邻两项求中点，作为边界点
#         w = [0] + list(ww[0]) + [tmp_frame[j].max()]  #把首末边界点加上
#         #df3 = pd.qcut(data,w,labels = range(k))
#         c,s = pd.qcut(tmp_frame[j].unique(), w, labels = range(k)) #使用自带的qcut功能切分
#==============================================================================
    c,s = pd.qcut(tmp_frame.iloc[:,0].unique(),20,retbins =1) #使用自带的qcut功能切分
    def get_group_num(x):
        for i in range(len(s-1)):
            if x<=s[i+1]:
                return i
    tmp_frame['group_num'] = tmp_frame.iloc[:,0].apply(get_group_num)

def tool_sas_rank(tmp_frame,group):
    '''
    这个按照 sas 公式实现rank分组功能，公式为
    floor(rank*k/(n+1))
    floor() 返回数字的下舍整数
    '''
    lenth = len(tmp_frame)
    tmp_frame['rank'] = tmp_frame.ix[:,0].rank(method='min')
    tmp_frame['group_num'] = tmp_frame.apply(lambda row : np.floor(row['rank']*group/(lenth+1)), axis=1)    
    
def tool_group_bygiven(tmp_frame,group):
    s = group['max']
    s.reset_index(drop=True,inplace=True)
    def get_group_num(x):
        for i in range(len(s)):
            if x<=s[i]:
                return i
    tmp_frame['group_num'] = tmp_frame.iloc[:,0].apply(get_group_num)


def cal_iv(data,group=20,ycol=-1):
    '''
    计算iv，默认用rank分组.
    对sas可能有误，需要核实。这边认为是根据值去重后排rank，再根据rank来等分。
    '''
    all_iv_detail = pd.DataFrame([])
    
    if type(ycol) == int:
        ycol = data.columns[ycol]
    
    if type(group) == int:
        column_names = data.columns[data.columns != ycol]
    elif isinstance(group,pd.DataFrame):
        column_names = group['var_name'].unique()
    else:
        print("argument 'group' type is wrong")
        return 0,0        

#    flag_ = 0    
    for i in column_names: #默认y在最后一列
        print(i)
        tmp = pd.concat([pd.DataFrame(data[i]),data[[ycol]]],axis=1)#tmp是临时的iv计算数据框
        tmp = tmp.astype('float')
        tmp.sort_values(by=tmp.columns[0],inplace=True)
        if type(group) == int:
            tool_sas_rank(tmp,group) #使用上面写的分组函数
        else:
            tool_group_bygiven(tmp,group[group['var_name']==i])
        grouped = tmp.groupby(tmp['group_num'])
        cols = grouped[tmp.columns[0]].agg({'min':min,'max':max})
        cols['group'] = range(len(cols))
        
        def len_minus_sum(x):
            ''' 默认了 1 代表坏人'''
            return len(x)-sum(x)
        
        col2 = grouped[tmp.columns[1]].agg({'y1_num':sum,'y0_num':len_minus_sum,'N':'size'})      
        cols = pd.concat([cols,col2],axis=1)
        bad_totl_num = float(tmp[tmp.columns[1]].sum())
        good_totl_num = float(len(tmp) - bad_totl_num)
        cols['y1_total'] = bad_totl_num
        cols['y0_total'] = good_totl_num
        cols['y1_percent'] = cols['y1_num'] / cols['y1_total']
        cols['y0_percent'] = cols['y0_num'] / cols['y0_total']
        cols['total_percent'] = cols['N'] / (cols['y1_total'] + cols['y0_total'])
        cols['WOE'] = np.log(cols['y0_percent']/cols['y1_percent'])
        cols.ix[cols['WOE'] == np.inf,'WOE'] = 0     # 分母为0的先设置为0吧
        cols.ix[cols['WOE'] == -np.inf,'WOE'] = 0    # 分母为0的先设置为0吧
        cols['MIV'] = (cols['y0_percent']-cols['y1_percent'])*cols['WOE']
        cols['ori_IV'] = cols['MIV'].sum()
        cols['var_name'] = i
        cols['bad_percent'] = cols['y1_num'] / cols['N']

        all_iv_detail = pd.concat([all_iv_detail,cols],axis=0)
        
#        flag_ = flag_+1
#        if flag_>3:
#            break
   
    all_iv_detail = all_iv_detail.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
    all_iv = all_iv_detail[['var_name','group','N','WOE','MIV','ori_IV', \
                            'y0_num','y1_num','min','max','bad_percent']]
    return all_iv, all_iv_detail  

    
def group_manual_regulation(group,iv_detail):
    #setp3 手工调整分组，重新计算iv
    #'''
    #group 是调整后的分组，是DataFrame格式。iv_detail是上一步返回的iv详细信息，也是df格式
    #这里的做法只要手工调整 iv.xls 的group分组就可以，重复执行这段代码直至满意
    #'''
    good_totl_num = iv_detail['y0_total'].iloc[0]
    bad_totl_num = iv_detail['y1_total'].iloc[0]
    pre_row_key = group.ix[0,:][['var_name','group']] #获取第一个值
    cols = pd.DataFrame()
    new_iv = pd.DataFrame()
    for row in group.iterrows():
        row = row[1]
        row_key = row[['var_name','group']]
        if (row_key != pre_row_key).any():
            if 'N_old' not in cols.columns:
                cols['N_old'] = cols['N']
                cols['WOE_old'] = cols['WOE']
                cols['y0_num_old'] = cols['y0_num']
                cols['y1_num_old'] = cols['y1_num']
                cols['MIV_old'] = cols['MIV']
            cols['y0_num'] = cols['y0_num_old'].sum()
            cols['y1_num'] = cols['y1_num_old'].sum()
            cols['N'] = cols['N_old'].sum()
            cols['y1_total'] = bad_totl_num
            cols['y0_total'] = good_totl_num
            cols['y1_percent'] = cols['y1_num'] / cols['y1_total']
            cols['y0_percent'] = cols['y0_num'] / cols['y0_total']
            cols['WOE'] = np.log(cols['y0_percent']/cols['y1_percent'])
            cols.ix[cols['WOE'] == np.inf,'WOE'] = 0 # 分母为0的先设置为0吧
            cols.ix[cols['WOE'] == -np.inf,'WOE'] = 0 # 分母为0的先设置为0吧
            cols['MIV'] = (cols['y0_percent']-cols['y1_percent'])*cols['WOE']
            new_iv = pd.concat((new_iv,cols),axis=0)
            cols = pd.DataFrame(row).T #用于存放一个新的组。这个新组只有在发现row_key不同的时候才触发
        elif (row_key == pre_row_key).all():
            cols = pd.concat((cols,pd.DataFrame(row).T),axis=0)
        else:
            print("Error: step3 unsuppose row")
        pre_row_key = row_key
        
        #最后一组放进来        
    if 'N_old' not in cols.columns:
        cols['N_old'] = cols['N']
        cols['WOE_old'] = cols['WOE']
        cols['y0_num_old'] = cols['y0_num']
        cols['y1_num_old'] = cols['y1_num']
        cols['MIV_old'] = cols['MIV']
    cols['y0_num'] = cols['y0_num_old'].sum()
    cols['y1_num'] = cols['y1_num_old'].sum()
    cols['N'] = cols['N_old'].sum()
    cols['y1_total'] = bad_totl_num
    cols['y0_total'] = good_totl_num
    cols['y1_percent'] = cols['y1_num'] / cols['y1_total']
    cols['y0_percent'] = cols['y0_num'] / cols['y0_total']
    cols['WOE'] = np.log(cols['y0_percent']/cols['y1_percent'])
    cols.ix[cols['WOE'] == np.inf,'WOE'] = 0 # 分母为0的先设置为0吧
    cols.ix[cols['WOE'] == -np.inf,'WOE'] = 0 # 分母为0的先设置为0吧
    cols['MIV'] = (cols['y0_percent']-cols['y1_percent'])*cols['WOE']
    new_iv = pd.concat((new_iv,cols),axis=0)
    cols = pd.DataFrame(row).T #用于存放一个组
    
    #计算新iv,先简单实现，唯一值相加，TODO：按group分组后相加
    new_iv['IV'] = 0
    for i in new_iv['var_name'].unique():
        tmp = new_iv[new_iv['var_name'] == i]
        tmp = tmp.groupby('group').max() #去重
        iv = tmp['MIV'].sum()
        new_iv.ix[new_iv['var_name'] == i,'IV'] = iv
        
    #    new_iv[new_iv['var_name'] == i]['IV'] = iv
        
    new_iv = new_iv[['var_name','group','N','N_old','WOE','WOE_old','MIV','MIV_old','ori_IV','IV', \
                     'y0_num','y0_num_old','y1_num','y1_num_old','min','max']] #调整顺序
    return new_iv
#    new_iv.to_excel(TOPIC_PATH+"/data/iv.xls")

def __replace_value(x,woe_frame):
    #回填变量工具，由于区间是逐渐增大的，如果在区间内，返回相应的woe
    for i in range(len(woe_frame)):
        if woe_frame['min'][i]<=x<= woe_frame['max'][i]:
            return woe_frame['WOE'][i]

def filling_woe(group,data):    
    for i in group['var_name'].unique():
        print(i)
        woe_frame = group[group['var_name'] == i][['min','max','WOE']]
        woe_frame.reset_index(drop=True,inplace=True)
        woe_frame = woe_frame.sort_values(by = 'max')
        data[i] = data[i].apply(__replace_value,args=(woe_frame,))
    return data



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
        woe_frame = woe[woe['var_name'] == i][['var_name','min','max','WOE']]
        beta_i = formular[formular[u"参数"] == i][u"估计值"].iloc[0]
        woe_frame['score'] = woe_frame['WOE'].apply(lambda woe : offset/n - factor*(a/n+beta_i*woe))
        scorecard = pd.concat((scorecard,woe_frame),axis=0)
        
    return scorecard
    
    
    
    
    