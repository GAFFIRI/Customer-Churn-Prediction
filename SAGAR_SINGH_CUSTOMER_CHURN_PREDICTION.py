#!/usr/bin/env python
# coding: utf-8

# In[335]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[336]:


df=pd.read_csv(r"C:\Users\Sagar\Downloads\Dataset.csv")


# In[337]:


df.head()


# In[ ]:





# In[338]:


print(df.dtypes)


# In[339]:


df.isnull().sum() # there is no null values in the data 


# In[340]:


df.groupby(['DataPlan'])['AccountWeeks'].mean()


# In[341]:


df.groupby(['ContractRenewal'])['AccountWeeks'].mean()


# In[342]:


df.groupby(['DataUsage'])['AccountWeeks'].mean()


# In[343]:


df.groupby(['MonthlyCharge'])['AccountWeeks'].mean()


# ## OUTLIER ANALYSIS

# In[344]:


df_1=['DataUsage','DayMins','MonthlyCharge','OverageFee','RoamMins']
#df_1.head()
#df_1=df_1.copy()


# In[345]:


import seaborn as sns 
for col in df_1:
    sns.boxplot(x=col, data=df)
    plt.show()


# we certainly have outliers in our data....
# handling outliers by flooring and capping

# In[346]:


for column in df_1:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    ll = q1 - (1.5 * iqr)
    ul = q3 + (1.5 * iqr)
    for index in df[column].index:
        if (df.loc[index, column] > ul).any():
            df.loc[index, column] = ul
        if (df.loc[index, column] < ll).any():
            df.loc[index, column] = ll


# In[347]:


for col in df_1:
    sns.boxplot(x=col, data=df)
    plt.show()


# # EDA

# univariant analysis

# In[348]:


con_col=['MonthlyCharge', 'DayMins', 'OverageFee','DataUsage','RoamMins']
cat_col = list(set(df.columns)-set(con_col))
print(cat_col)


# In[349]:


# for continuos data we use hitogram and kde plots lets see them


# In[350]:


import matplotlib.pyplot as plt
for col in con_col:
    sns.kdeplot(x=col,data=df)
    plt.show()


# In[351]:


for col in con_col:
    print(col,':',df[col].skew())


# # as u can see all the con_col except DataUsage are normaly distributed.
# # data usage are higly right skiewed. DataUsage : 1.2720573148196221

# In[352]:


# for categorical data we use bar plots lets see them


# In[353]:


for col in cat_col:
    plt.figure(figsize=(30,8))
    sns.countplot(x=col,data=df)
    plt.show()


# #inferences 
# # most of the customer have data plans 
# # most of the customer hv recently renewed contract
# # most  of the customer only get 1 - 4 customer service call 
# # % of churned customer are less about 14.49%

# In[354]:


df["Churn"].value_counts()/len(df['Churn'])*100


# ### Bivarient Analysis
# target variable vs. independent variable

# In[355]:


for col in con_col:
    sns.boxplot(x=df['Churn'],y=col,data=df)
    plt.show()


# In[356]:


#inferences 
# Daymins are higher of churned customers
# datausage of nonchurned customer are high.
# roammins , overaage fee and monthly charges has no impact no 


# In[357]:


# target vs categorical data 
# categorical vs. categorical data
# we use stacked bar chart 


# In[358]:


for col in cat_col:
    #plt.figure(figsize)
    pd.crosstab(df[col],df['Churn']).plot(kind='bar',figsize=(20,10))
    plt.show()


# In[359]:


# Churned customer has less dataplans as compared to (class 0)
# churned customer have more contract renewal


# ## Scaling - continuos data

# In[360]:


from sklearn.preprocessing import StandardScaler


# In[361]:


ss = StandardScaler()
num_cols = con_col + ['AccountWeeks','DayCalls','CustServCalls']
x_con_scaled = pd.DataFrame(ss.fit_transform(df[num_cols]),columns=num_cols,index=df.index)


# In[362]:


x_con_scaled


# # ENCODING 

# categorical data into numerical data using one hot encoding.

# In[363]:


cat_col


# In[364]:


cat_col


# In[365]:


cat_col.remove('AccountWeeks')
cat_col.remove('CustServCalls')
cat_col.remove('DayCalls')
cat_col.remove('Churn')


# In[366]:


cat_col


# In[367]:


#df['ContractRenewal']= df['ContractRenewal'].astype(object)
#df['Churn']= df['Churn'].astype(object)
#df['DataPlan']= df['DataPlan'].astype(object)


# In[368]:


x_cat_enc = pd.get_dummies(df[cat_col],drop_first=True)


# In[371]:


x_final = pd.concat([x_con_scaled,x_cat_enc],axis=1)
x_final.head()


# ## train - test split

# In[372]:


from sklearn.model_selection import train_test_split


# In[373]:


y =  df['Churn']


# Training - 80%, Testing = 20% 

# In[375]:


x_train,x_test,y_train,y_test = train_test_split(x_final,y,test_size=0.2,random_state=42)


# In[376]:


df.head()


# # Logistic model 

# In[377]:


from sklearn.linear_model import LogisticRegression


# In[378]:


log_reg = LogisticRegression(penalty='l2',C=1.0,class_weight='balanced',fit_intercept=True)
log_reg.fit(x_train,y_train)


# In[379]:


print('Intercept:',log_reg.intercept_)
print('Coefficients:',log_reg.coef_[0])


# # Performance matrix 

# ## CONFUSION MATRIX

# In[380]:


y_test_pred = log_reg.predict(x_test)


# In[381]:


from sklearn.metrics import confusion_matrix,classification_report


# In[382]:


sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")


# In[383]:


print(classification_report(y_test,y_test_pred))


# Judging by the F1_score, our model is giving us : 79% accuracy

# True Positives (TP): The number of customers correctly predicted as churned. 
# customers who have actually canceled the service, model correctly identifies them as churned.
# 
# False Negatives (FN): The number of customers wrongly predicted as non-churned.
# customers who have actually canceled the service and model incorrectly identifies them as non-churned
#  
# churner prediction in the light of bussiness requirement should be focusing on maximizing TP and minimizing FN

# In[ ]:


# f1 score of positive class is not much better it is only 53%, for a good model both presicion and recall should be high


# ## ROC-CuRVE

# In[395]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[396]:


y_test_pred_prob = log_reg.predict_proba(x_test)[:,1]
fpr,tpr,thresh = roc_curve(y_test,y_test_pred_prob,drop_intermediate=True)# y_pred_probability  
roc_df = pd.DataFrame({'TPR':tpr,"FPR":fpr,'Threshold':thresh})

roc_df


# In[397]:


plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(roc_df['FPR'],roc_df['TPR'])
plt.plot([[0,0],[1,1]],color='red')
plt.show()


# In[ ]:


# best threshold : 1.981544


# # train-test score

# In[385]:


y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)


# In[386]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# ### for classification report 

# In[394]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("*"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# In[398]:


#recall score is 74% inn train , and 80% in test , very less impact of overfitting


# In[399]:


# lets check the train and test ROC 


# In[401]:


y_train_pred_pob =log_reg.predict_proba(x_train)[:,1]
y_test_pred_pob =log_reg.predict_proba(x_test)[:,1]


# In[402]:


print('train_roc:',roc_auc_score(y_train,y_train_pred_pob))
print('test_roc:',roc_auc_score(y_test,y_test_pred_pob))


# In[ ]:


#train and test roc_score are very clsoe to each other, which mean there is no overfitting in the model


# # Cross validation score 

# In[388]:


from sklearn.model_selection import cross_val_score


# In[391]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='recall',cv=4)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# In[ ]:


# our model is having constant performance with average performance is around 73% 


# In[403]:


# checking cross validation for roc


# In[404]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='roc_auc',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# almost similar roc_suc score in every subset with avdg score of 0.8115256312653527

# # Finding best threshold by ROC 

# for good model our TPR(True Positive Recall) should be high and FPR(False Positive Recall) should be low.
# >>we can interpret by it tht difference betwwen the TPR  and FPR should be high. 

# In[412]:


roc_df['Difference'] = roc_df['TPR']-roc_df['FPR']
roc_df[roc_df['Difference']==max(roc_df['Difference'])]


# In[413]:


#0.510122 is the best threshold for this data... which is  almost closer to 0.5 


# In[ ]:


# there is not much differnce between precision and recall  so i doubt tht class imblance is paying role here 


# # Hyperparameter tunning for Logistic Regression

# In[415]:


from sklearn.model_selection import GridSearchCV


# In[416]:


LogisticRegression()


# In[417]:


grid = {'penalty' : [ 'l2', 'none'],
        'C' : np.arange(0.1,3,0.5)
       }


# In[418]:


grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'),grid,scoring='recall',cv=5)
grid_search.fit(x_train,y_train)


# In[421]:


grid_search.best_params_


# In[422]:


pd.DataFrame(grid_search.cv_results_)


# #### BUilding the model with updated parameter {'C': 0.1, 'penalty': 'l2'}

# In[423]:


from sklearn.linear_model import LogisticRegression


# In[424]:


log_reg = LogisticRegression(penalty='l2',C=0.1,class_weight='balanced',fit_intercept=True)
log_reg.fit(x_train,y_train)


# In[425]:


print('Intercept:',log_reg.intercept_)
print('Coefficients:',log_reg.coef_[0])


# # train-test score

# In[426]:


y_train_pred = log_reg.predict(x_train)
y_test_pred = log_reg.predict(x_test)


# In[427]:


print('Train confusion Matrix:')
sns.heatmap(confusion_matrix(y_train,y_train_pred),annot=True,fmt=".0f")
plt.show()
print('Test confusion Matrix:')
sns.heatmap(confusion_matrix(y_test,y_test_pred),annot=True,fmt=".0f")
plt.show()


# ### for classification report 

# In[428]:


print('Train Classification Report:')
print(classification_report(y_train,y_train_pred))
print("*"*100)
print('Test Classification Matrix:')
print(classification_report(y_test,y_test_pred))


# In[398]:


#recall score is 74% inn train , and 81% in test ,not a much improvement than the previous model


# In[399]:


# lets check the train and test ROC 


# In[429]:


y_train_pred_pob =log_reg.predict_proba(x_train)[:,1]
y_test_pred_pob =log_reg.predict_proba(x_test)[:,1]


# In[430]:


print('train_roc:',roc_auc_score(y_train,y_train_pred_pob))
print('test_roc:',roc_auc_score(y_test,y_test_pred_pob))


# In[431]:


#train and test roc_score are very clsoe to each other


# # Cross validation score 

# In[432]:


from sklearn.model_selection import cross_val_score


# In[434]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='recall',cv=4)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# In[435]:


# our model is having constant performance with average performance is around 74% 


# In[436]:


# checking cross validation for roc


# In[437]:


scores = cross_val_score(log_reg,x_train,y_train,scoring='roc_auc',cv=5)
print('Score:',scores)
print('Avg Score:',np.mean(scores))
print('Std Score:',np.std(scores))


# almost similar roc_suc score in every subset with avdg score of 0.8115256312653527

# In[ ]:





# In[438]:


print('Intercept:',log_reg.intercept_)
print('Coefficients:',log_reg.coef_[0])


# In[439]:


co_ef_df = pd.DataFrame({'Colums':x_final.columns,'Co_eff':log_reg.coef_[0]})
co_ef_df['exp_coef'] = np.exp(co_ef_df['Co_eff'])
co_ef_df


# ### if datausage increases by 1 unit then churned customer decreases by 32% 
# # if ContractRenewal increases by 1 unit then churned customer decreases by 86% 
# # if DataPlan increases by 1 unit then churned customer decreases by 34% 
# # if AccountWeeks increases by 1 unit then churned customer increases by 4% 
# # if MonthlyCharge increases by 1 unit then churned customer increases by 30%
# # if DayMins increases by 1 unit then churned customer increases by 61%
# # if OverageFee increases by 1 unit then churned customer increases by 20%
# 

# ## conclusion : acc to mine domian knowledge our model is handleling data pretty well with 74% accuracy we dont need to do the class imbalance handeling as it performing well.hence compnay should focus on the above explained feature to controll the churning of thier customer
# 
# 
