library('smbinning')  #最优分箱
library('DMwR')  #检测离群值
library('xlsx')  

####################################################################################
###客户基本信息 和 征信数据衍生变量
#readFilePath<-"F:/TS/Lending_Club/05_middle/data_loan.csv"
readFilePath<-"C:/Users/Administrator/Desktop/df7.csv"
df<-read.csv(readFilePath)
head(df)
names(df)


#smbinning(df, y, x, p = 0.05)
#df: 数据
#y： 二分类变量(0,1) 整型
#x：连续变量：至少满足10 个不同值，取值范围有限
#p：每个Bin记录数占比，默认5% (0.05) 范围0%-50%
#smbinning.plot, smbinning.sql,and smbinning.gen.


result1<-smbinning(df=df,x="acc_open_past_24mths",y="y",p=0.05)
smbinning.plot(result1,option="WoE",sub="acc_open_past_24mths")
r1 <- merge(result1$x,result1$ivtable)

result2<-smbinning(df=df,x="inq_last_12m",y="y",p=0.05)
smbinning.plot(result2,option="WoE",sub="inq_last_12m")
r2 <- merge(result2$x,result2$ivtable)

result3<-smbinning(df=df,x="bc_open_to_buy",y="y",p=0.05)
smbinning.plot(result3,option="WoE",sub="bc_open_to_buy")
r3 <- merge(result3$x,result3$ivtable)

result4<-smbinning(df=df,x="mths_since_rcnt_il",y="y",p=0.05)
smbinning.plot(result4,option="WoE",sub="mths_since_rcnt_il")
r4 <- merge(result4$x,result4$ivtable)

result5<-smbinning(df=df,x="mo_sin_rcnt_tl",y="y",p=0.05)
smbinning.plot(result5,option="WoE",sub="mo_sin_rcnt_tl")
r5 <- merge(result5$x,result5$ivtable)

result6<-smbinning(df=df,x="open_il_12m",y="y",p=0.05)
smbinning.plot(result6,option="WoE",sub="open_il_12m")
r6 <- merge(result6$x,result6$ivtable)

result7<-smbinning(df=df,x="all_util",y="y",p=0.05)
smbinning.plot(result7,option="WoE",sub="all_util")
r7 <- merge(result7$x,result7$ivtable)

result8<-smbinning(df=df,x="open_acc_6m",y="y",p=0.05)
smbinning.plot(result8,option="WoE",sub="open_acc_6m")
r8 <- merge(result8$x,result8$ivtable)

result9<-smbinning(df=df,x="dti",y="y",p=0.05)
smbinning.plot(result9,option="WoE",sub="dti")
r9 <- merge(result9$x,result9$ivtable)

result10<-smbinning(df=df,x="il_util",y="y",p=0.05)
smbinning.plot(result10,option="WoE",sub="il_util")
r10 <- merge(result10$x,result10$ivtable)

result11<-smbinning(df=df,x="mo_sin_rcnt_rev_tl_op",y="y",p=0.05)
smbinning.plot(result11,option="WoE",sub="mo_sin_rcnt_rev_tl_op")
r11 <- merge(result11$x,result11$ivtable)

result12<-smbinning(df=df,x="inq_fi",y="y",p=0.05)
smbinning.plot(result12,option="WoE",sub="inq_fi")
r12 <- merge(result12$x,result12$ivtable)

result13<-smbinning(df=df,x="tot_hi_cred_lim",y="y",p=0.05)
smbinning.plot(result13,option="WoE",sub="tot_hi_cred_lim")
r13 <- merge(result13$x,result13$ivtable)

result14<-smbinning(df=df,x="mort_acc",y="y",p=0.05)
smbinning.plot(result14,option="WoE",sub="mort_acc")
r14 <- merge(result14$x,result14$ivtable)

result15<-smbinning(df=df,x="mths_since_recent_inq",y="y",p=0.05)
smbinning.plot(result15,option="WoE",sub="mths_since_recent_inq")
r15 <- merge(result15$x,result15$ivtable)

result16<-smbinning(df=df,x="percent_bc_gt_75",y="y",p=0.05)
smbinning.plot(result16,option="WoE",sub="percent_bc_gt_75")
r16 <- merge(result16$x,result16$ivtable)

r_total <- rbind(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16)
outFilePath <- "F:/TS/Lending_Club/04_output/03_r_smbining/r_best_binging.xlsx"
write.xlsx(r_total, outFilePath)  

####################################################################################
# Information Value for all variables in one step ---------------------------
smbinning.sumiv(df=df,y="y") # IV for eache variable

# Plot IV for all variables -------------------------------------------------
sumivt=smbinning.sumiv(df,y="y")
sumivt # Display table with IV by characteristic
par(mfrow=c(1,1))
smbinning.sumiv.plot(sumivt,cex=1) # Plot IV summary table

####################################################################################











library(smbinning) 
# Run and save results 
result=smbinning(df=df,y="y",x="score",p=0.05) 
result$ivtable

# Relevant plots (2x2 Page) 
par(mfrow=c(2,2)) 
boxplot(df$score~df$y, 
        horizontal=T, frame=F, col="lightgray",main="Distribution") 
mtext("score",3) 
smbinning.plot(result,option="dist",sub="score") 
smbinning.plot(result,option="badrate",sub="score") 
smbinning.plot(result,option="WoE",sub="score")

####################################################################################
##第二种方法计算IV值
library(devtools)  
library(woe)   
df1 <- df
df1$y<-as.factor(df1$y)  # 变量类型修改，目标变量要改成factor类型

df1$education_g <-as.character(df1$education_g) # 其他的输入变量可以直接修改成 character类型

IV <- iv.mult(df1$education_g, df1$y)  #原理是以Y作为被解释变量，其他作为解释变量，建立决策树模型  
iv.plot.summary(IV)  



age_woe <- woe(df1,"age",Continuous=F,"y",C_Bin=4,Good="1",Bad="0")
ggplot(age_woe,aes(x=BIN,y=-WOE)) + 
  geom_bar(stat="identity",fill="blue",colour="grey60",size=0.2,alpha=0.2)+labs(title="age")


####################################################################################
##只有customer_info的数据
readFilePath<-"F:/TS/offline_model/01_Dataset/02_Interim/middle_data/middle_data2.csv"
df<-read.csv(readFilePath)

head(df)   #显示数据框Data的前几行。
names(df)  #显示数据框Data的变量名。
df$name1   #数据框Data中名为name1的变量。

smbinning.factor(df = df, y = "y", x = "education_g")
smbinning.sql(result2)

result1<-smbinning(df=df,x="age",y="y",p=0.05)
smbinning.plot(result1,option="WoE",sub="age")

result2<-smbinning(df=df,x="yearly_income",y="y",p=0.05)
smbinning.plot(result2,option="WoE",sub="yearly_income")
# outFilePath <- "F:/TS/offline_model/01_Dataset/04_Output/woe_r/yearly_income.xlsx"
# write.xlsx(result$ivtable, outFilePath)  
# write.csv(result$ivtable, outFilePath, row.names = "yearly_income", quote = F)

result3<-smbinning(df=df,x="monthly_salary",y="y",p=0.05)
smbinning.plot(result3,option="WoE",sub="monthly_salary")

result4<-smbinning(df=df,x="monthly_other_income",y="y",p=0.05)
smbinning.plot(result4,option="WoE",sub="monthly_other_income")

result5<-smbinning(df=df,x="credit_amt_all",y="y",p=0.05)
smbinning.plot(result5,option="WoE",sub="credit_amt_all")

result6<-smbinning(df=df,x="credit_use_ratio",y="y",p=0.05)
smbinning.plot(result6,option="WoE",sub="credit_use_ratio")

result7<-smbinning(df=df,x="monthly_salary",y="y",p=0.05)
smbinning.plot(result7,option="WoE",sub="monthly_salary")

result8<-smbinning(df=df,x="debt_all",y="y",p=0.05)
smbinning.plot(result8,option="WoE",sub="debt_all")

result9<-smbinning(df=df,x="desired_loan_amount",y="y",p=0.05)
smbinning.plot(result9,option="WoE",sub="desired_loan_amount")

result10<-smbinning(df=df,x="loan_month_return_new",y="y",p=0.05)
smbinning.plot(result10,option="WoE",sub="loan_month_return_new")

result11<-smbinning(df=df,x="score",y="y",p=0.05)
smbinning.plot(result11,option="WoE",sub="score")


result12<-smbinning(df=df,x="selfquery_cardquery_in6m",y="y",p=0.05)
smbinning.plot(result12,option="WoE",sub="selfquery_cardquery_in6m")

result13<-smbinning(df=df,x="card_60_pastdue_frequency",y="y",p=0.05)
smbinning.plot(result13,option="WoE",sub="card_60_pastdue_frequency")

result14<-smbinning(df=df,x="self_query_03_month_frequency",y="y",p=0.05)
smbinning.plot(result14,option="WoE",sub="self_query_03_month_frequency")

result15<-smbinning(df=df,x="card_apply_03_month_frequency",y="y",p=0.05)
smbinning.plot(result15,option="WoE",sub="card_apply_03_month_frequency")

result16<-smbinning(df=df,x="max_cardline",y="y",p=0.05)
smbinning.plot(result16,option="WoE",sub="max_cardline")

result17<-smbinning(df=df,x="self_query_24_month_frequency",y="y",p=0.05)
smbinning.plot(result17,option="WoE",sub="self_query_24_month_frequency")
####################################################################################



# Information Value for all variables in one step ---------------------------
smbinning.sumiv(df=df,y="y") # IV for eache variable

# Plot IV for all variables -------------------------------------------------
sumivt=smbinning.sumiv(df,y="y")
sumivt # Display table with IV by characteristic
par(mfrow=c(1,1))
smbinning.sumiv.plot(sumivt,cex=1) # Plot IV summary table












readFilePath<-"C:/Users/Administrator/Desktop/df7.csv"
df<-read.csv(readFilePath)
head(df)
names(df)


#smbinning(df, y, x, p = 0.05)
#df: 数据
#y： 二分类变量(0,1) 整型
#x：连续变量：至少满足10 个不同值，取值范围有限
#p：每个Bin记录数占比，默认5% (0.05) 范围0%-50%
#smbinning.plot, smbinning.sql,and smbinning.gen.


result1<-smbinning(df=df,x="int_rate",y="y",p=0.05)
smbinning.plot(result1,option="WoE",sub="int_rate")
r1 <- merge(result1$x,result1$ivtable)

result2<-smbinning(df=df,x="acc_open_past_24mths",y="y",p=0.05)
smbinning.plot(result2,option="WoE",sub="acc_open_past_24mths")
r2 <- merge(result2$x,result2$ivtable)

result3<-smbinning(df=df,x="inq_last_12m",y="y",p=0.05)
smbinning.plot(result3,option="WoE",sub="inq_last_12m")
r3 <- merge(result3$x,result3$ivtable)

result4<-smbinning(df=df,x="num_tl_op_past_12m",y="y",p=0.05)
smbinning.plot(result4,option="WoE",sub="num_tl_op_past_12m")
r4 <- merge(result4$x,result4$ivtable)

result5<-smbinning(df=df,x="mths_since_rcnt_il",y="y",p=0.05)
smbinning.plot(result5,option="WoE",sub="mths_since_rcnt_il")
r5 <- merge(result5$x,result5$ivtable)

result6<-smbinning(df=df,x="open_rv_24m",y="y",p=0.05)
smbinning.plot(result6,option="WoE",sub="open_rv_24m")
r6 <- merge(result6$x,result6$ivtable)

result7<-smbinning(df=df,x="mo_sin_rcnt_tl",y="y",p=0.05)
smbinning.plot(result7,option="WoE",sub="mo_sin_rcnt_tl")
r7 <- merge(result7$x,result7$ivtable)

result8<-smbinning(df=df,x="open_il_24m",y="y",p=0.05)
smbinning.plot(result8,option="WoE",sub="open_il_24m")
r8 <- merge(result8$x,result8$ivtable)

result9<-smbinning(df=df,x="open_il_12m",y="y",p=0.05)
smbinning.plot(result9,option="WoE",sub="open_il_12m")
r9 <- merge(result9$x,result9$ivtable)

result10<-smbinning(df=df,x="open_acc_6m",y="y",p=0.05)
smbinning.plot(result10,option="WoE",sub="open_acc_6m")
r10 <- merge(result10$x,result10$ivtable)

result11<-smbinning(df=df,x="dti",y="y",p=0.05)
smbinning.plot(result11,option="WoE",sub="dti")
r11 <- merge(result11$x,result11$ivtable)

result12<-smbinning(df=df,x="all_util",y="y",p=0.05)
smbinning.plot(result12,option="WoE",sub="all_util")
r12 <- merge(result12$x,result12$ivtable)

result13<-smbinning(df=df,x="il_util",y="y",p=0.05)
smbinning.plot(result13,option="WoE",sub="il_util")
r13 <- merge(result13$x,result13$ivtable)

result14<-smbinning(df=df,x="mo_sin_rcnt_rev_tl_op",y="y",p=0.05)
smbinning.plot(result14,option="WoE",sub="mo_sin_rcnt_rev_tl_op")
r14 <- merge(result14$x,result14$ivtable)

result15<-smbinning(df=df,x="mths_since_recent_inq",y="y",p=0.05)
smbinning.plot(result15,option="WoE",sub="mths_since_recent_inq")
r15 <- merge(result15$x,result15$ivtable)

result16<-smbinning(df=df,x="inq_fi",y="y",p=0.05)
smbinning.plot(result16,option="WoE",sub="inq_fi")
r16 <- merge(result16$x,result16$ivtable)

result17<-smbinning(df=df,x="open_rv_12m",y="y",p=0.05)
smbinning.plot(result17,option="WoE",sub="open_rv_12m")
r17 <- merge(result17$x,result17$ivtable)

result18<-smbinning(df=df,x="total_rec_int",y="y",p=0.05)
smbinning.plot(result18,option="WoE",sub="total_rec_int")
r18 <- merge(result18$x,result18$ivtable)

result19<-smbinning(df=df,x="mths_since_recent_bc",y="y",p=0.05)
smbinning.plot(result19,option="WoE",sub="mths_since_recent_bc")
r19 <- merge(result19$x,result19$ivtable)

result20<-smbinning(df=df,x="tot_cur_bal",y="y",p=0.05)
smbinning.plot(result20,option="WoE",sub="tot_cur_bal")
r20 <- merge(result20$x,result20$ivtable)

result21<-smbinning(df=df,x="avg_cur_bal",y="y",p=0.05)
smbinning.plot(result21,option="WoE",sub="avg_cur_bal")
r21 <- merge(result21$x,result21$ivtable)

result22<-smbinning(df=df,x="tot_cur_bal",y="y",p=0.05)
smbinning.plot(result22,option="WoE",sub="tot_cur_bal")
r22 <- merge(result22$x,result22$ivtable)

result23<-smbinning(df=df,x="mort_acc",y="y",p=0.05)
smbinning.plot(result23,option="WoE",sub="mort_acc")
r23 <- merge(result23$x,result23$ivtable)

result24<-smbinning(df=df,x="total_rev_hi_lim",y="y",p=0.05)
smbinning.plot(result24,option="WoE",sub="total_rev_hi_lim")
r24 <- merge(result24$x,result24$ivtable)

r_total <- rbind(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24)
outFilePath <- "F:/TS/Lending_Club/04_output/03_r_smbining/r_best_binging01.xlsx"
write.xlsx(r_total, outFilePath)  