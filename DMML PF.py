#!/usr/bin/env python
# coding: utf-8

# Data Set Information:
# Seven different types of dry beans were used in this research, taking into account the features such as form, shape, type, and structure by the market situation.
# 
# A computer vision system was developed to distinguish seven different registered varieties of dry beans with similar features in order to obtain uniform seed classification.
# 
# For the classification model, images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera.
# 
# Bean images obtained by computer vision system were subjected to segmentation and feature extraction stages, and a total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains.

# Attribute Information:
# Bean Id (B): The counter of the rows
# 
# Area (A): The area of a bean zone and the number of pixels within its boundaries.
# 
# Perimeter (P): Bean circumference is defined as the length of its border.
# 
# Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
# 
# Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
# 
# Aspect ratio (K): Defines the relationship between L and l.
# 
# Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
# 
# Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
# 
# Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
# 
# Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
# 
# Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
# 
# Roundness (R): Calculated with the following formula: (4piA)/(P^2)
# 
# Compactness (CO): Measures the roundness of an object: Ed/L
# 
# ShapeFactor1 (SF1)
# 
# ShapeFactor2 (SF2)
# 
# ShapeFactor3 (SF3)
# 
# ShapeFactor4 (SF4)
# 
# Class: (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira)

# # IMPORT MODULES

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


# # LOADING THE DATASET

# In[4]:


df=pd.read_excel('Dry_Bean_Dataset.xlsx')
df = df.rename(columns={'roundness':'Roundness'})
#so every column name will start with capital letters and drop the Bean ID,because no information of Bean ID is related


# In[5]:


df.head()
#so we can observe the matrix


# In[6]:


df.tail()


# In[7]:


#to display descriptive statistics
df.describe()


# In[13]:


#to show basic info about datatype
df.info()


# In[14]:


#to display number of samples on each class
df['Class'].value_counts()


# # PREPROCESSING THE DATASET

# In[12]:


#check for null values
df.isnull().sum()


# # EXPLORATORY DATA ANALYSIS

# In[15]:


#histograms
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[16]:


#scatterplot, 
df.plot(kind="scatter", x="Compactness", y="ShapeFactor3",figsize=(10,8),alpha=0.1);


# ShapeFactor 3 is analogous to Compactness, there are no outliers on this scatterplot which shows the strong correlation between ShapeFactor3 and Compactness.

# In[17]:


df.plot(kind="scatter", x="Area", y="ConvexArea",figsize=(10,8),alpha=0.1);


# There are a few outliers on this graph, again meaning there is a strong correlation between ConvexArea and Area and they both share more values < 100000.
# 
# 

# In[18]:


df.plot(kind="scatter", x="EquivDiameter", y="Roundness",figsize=(10,8),alpha=0.1);


# The region Where Roundness is >0.8 and EquivDiameter is <350 is the most densly populated area in the graph.

# In[19]:


df.plot(kind="scatter", x="Area", y="Perimeter",figsize=(10,8),alpha=0.1);


# More points lie in the region where area < 100000 and perimeter is <1200, there is a strong correlation between these values. Out with this region in the graph there is still a strong correlation between values however it is not as densly populated. There are no major outlier on the graph.
# 
# 

# In[20]:


df.plot(kind="scatter", x="MajorAxisLength", y="Perimeter",figsize=(10,8));


# In[21]:


df.plot(kind="scatter", x="MajorAxisLength", y="MinorAxisLength", alpha=0.4,
    s=df["Perimeter"]/100, label="Perimeter", figsize=(10,10),
    c="Area", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend();


# As the area is smaller the graph is more populated. There is still a clear correlation between MinorAxisLength and MajorAxisLength however it is not as strong as previous graphs. The correlation becomes less clear where there is a higher value for area, although it is still slighlty visible.

# In[22]:


colors = ['red','blue','orange','purple','black','pink','green']
species= ['DERMASON','SIRA','SEKER','HOROZ','CALI','BARBUNYA','BOMBAY']
for i in range(7):
    x=df[df['Class'] == species[i]]
    plt.scatter(x['MajorAxisLength'],x['MinorAxisLength'], c=colors[i],label=species[i],s=100)
plt.xlabel("MajorAxisLength")
plt.ylabel("MinorAxisLength")
plt.legend();


# As we can see the barbunya is quite similar to cali, and the sira is quite similar to dermanson,seker and horoz. Bombay on the other hand is clusted far away from the other species.
# 
# The following is a correlation matrix where we are looking for the highest values to show strongest correlation

# In[23]:


corr = df.corr()
fig, ax=plt.subplots(figsize=(15,7))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm');


# The largest correlation is between Compactess and ShapeFactor3, and Area and ConvexArea where both are equal to 1. The second highest correlation is between equivdiameter and perimeter and is equal to 99. The third highest correlations are two the MajorAxislength with the Perimeter,and the Area with the Equivdiameter are equal to 98.

# # LABEL ENCODER

# In[24]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])
df.head(10)


# All Species converted to numbers.

# In[25]:


df['Class'].value_counts()


# In[26]:


corr = df.corr()
corr["Class"].sort_values(ascending=False)


# The above are used in the following new datasets.

# In[27]:


dataset1=df[["ShapeFactor1","Roundness"]]
dataset2=df[["ShapeFactor1","Roundness","ShapeFactor2","Solidity","ShapeFactor3"]]
dataset3=df[["ShapeFactor1","Roundness","ShapeFactor2","Solidity","ShapeFactor3","ShapeFactor4","Compactness","Extent","AspectRation","Eccentricity"]]


# I divided my data into 3 different data sets based on how related it is to the species. Ιn the beginning I got the first 2 attributes, then the first 5 and then the first 10.

# # MODEL TRAINING~

# In[28]:


#for my original data


# In[29]:


from sklearn.model_selection import train_test_split
# train-70% of the data
# test -30% of the data
X = df.drop(columns=['Class'])
Y = df['Class']
x_train, x_test, y_train,y_test=train_test_split(X,Y,test_size=.30)


# In[30]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print(y_pred)


# We see some of the predictions.

# In[31]:


from sklearn.metrics import confusion_matrix ,classification_report
cm=(np.array(confusion_matrix(y_test,y_pred)))
cm


# From the matrix we perceive that the values of diagonal are the correct predicts and the rest values of the matrix are the incorrects
# 
# We can also see that in the first row the correct vaules are almost the same with the incorrects. We can also see that in the second row there are only correct values.

# In[33]:


print(classification_report(y_test,y_pred))


# The Accuracy is equal to 77% and we can also see the precision,recall, f1-score and of BOMBAY(1) is equal to 1,which is the best score,because there are no incorrect values.
# 
# In the first row the percentages are low,because of the many incorrect values.

# In[34]:


TP = np.diag(cm)
TP


# True Positive values

# In[36]:


FP = cm.sum(axis=0) - np.diag(cm)
FP


# False Positive values

# # For my first dataset(dataset1)

# In[37]:


A=dataset1
Y = df['Class']
a_train, a_test, y1_train, y1_test=train_test_split(A,Y,test_size=.30)


# In[38]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb=GaussianNB()
gnb.fit(a_train,y1_train)
y_pred1=gnb.predict(a_test)
print(y_pred1)


# In[39]:


from sklearn.metrics import confusion_matrix ,classification_report
cm1=np.array(confusion_matrix(y1_test,y_pred1))
cm1


# From the matrix we perceive that the values of diagonal are the correct predicts and the rest values of the matrix are the incorrects. Also can be noticed that in the first row there are still many incorrect values.

# In[40]:


print(classification_report(y1_test,y_pred1))


# The accuracy is higher than my normal data(85%),because only positive correlations were used.In my normal data I have 9 negative correlations and 7 positive.Here I have only 2 positives. We can also see the precision,recall, f1-score and of BOMBAY(1) is equal to 1,which is the best score,because there are no incorrect values.

# In[41]:


FP = cm1.sum(axis=0) - np.diag(cm1)
FP


# False Positive values

# In[43]:


TP = np.diag(cm1)
TP


# True Positive values
# 
# 

# # For my second dataset2

# In[44]:


B=dataset2
Y = df['Class']
b_train, b_test, y2_train, y2_test=train_test_split(B,Y,test_size=.30)


# In[45]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb=GaussianNB()
gnb.fit(b_train,y2_train)
y_pred2=gnb.predict(b_test)
print(y_pred2)


# In[46]:


from sklearn.metrics import confusion_matrix ,classification_report
cm2=np.array(confusion_matrix(y2_test,y_pred2))
cm2


# From the matrix we perceive that the values of diagonal are the correct predicts and the rest values of the matrix are the incorrects. Also can be noticed that in the first row the incorrects values have been discreased and in the second row we have only one incorrect value.

# In[47]:


print(classification_report(y2_test,y_pred2))


# The accuracy is the highest in my second dataset,because more positive attirbutes were used and no negatives.

# We can also see the precision, f1-score and of BOMBAY(1) is equal to 1,which is the best score,because there is only one incorrect value and recall is equal to 99%.

# In[49]:


FP = cm2.sum(axis=0) - np.diag(cm2)
FP


# False Positive values
# 
# 

# In[50]:


TP = np.diag(cm2)
TP


# True Positive values
# 
# 

# # For my third dataset3
# 

# In[51]:


C=dataset3
Y = df['Class']
c_train, c_test, y3_train, y3_test=train_test_split(C,Y,test_size=.30)


# In[52]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb=GaussianNB()
gnb.fit(c_train,y3_train)
y_pred3=gnb.predict(c_test)
print(y_pred3)


# In[53]:


from sklearn.metrics import confusion_matrix ,classification_report
cm3=np.array(confusion_matrix(y3_test,y_pred3))
cm3


# From the matrix we perceive that the values of diagonal are the correct predicts and the rest values of the matrix are the incorrects. The most incorrect values are in the last row.

# In[54]:


print(classification_report(y3_test,y_pred3))


# The accuracy is 87%,which is higher than my normal data,because less negative attributes were used. It is also lower than my second dataset,because my second dataset has only positive attributes and no negatives.
# 
# The last row has the lowest percentage,because it has also the most incorrect values.

# In[55]:


FP = cm3.sum(axis=0) - np.diag(cm3)
FP


# False Positive values
# 
# 

# In[56]:


TP = np.diag(cm3)
TP


# True Positive values
# 
# So the best model is in dataset2,because it has the highest accuracy and has not negative correlated attributes with the Class. The second is in dataset3,because it has 7 positive and 3 negative correlated attributes with the Class. The third is in dataset1,which has only 2 positive correlated attibutes with the Class. The last one is in normal dataset,which has 7 positive and 9 negative correlated attributes with the Class.

# In[57]:


#decision tree
from sklearn.tree import DecisionTreeClassifier,plot_tree
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test, y_test))


# In[58]:


#Tree1
from sklearn import preprocessing

x=df[['Perimeter','MajorAxisLength','MinorAxisLength','EquivDiameter','ShapeFactor1']]
y=df['Class']
xt1_train, xt1_test, yt1_train,yt1_test=train_test_split(x,y,test_size=.30, random_state=101)
d_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
d_tree.fit(xt1_train,yt1_train)
print(d_tree.score(xt1_test, yt1_test))
plot_tree(d_tree)


# In[59]:


#Tree2
from sklearn import preprocessing

x1=df[['Perimeter','MajorAxisLength','MinorAxisLength','EquivDiameter','ShapeFactor1']]
y2=df['Class']
xt2_train, xt2_test, yt2_train,yt2_test=train_test_split(x1,y2,test_size=.30, random_state=101)
d_tree2 = DecisionTreeClassifier(max_depth=2, random_state=42)
d_tree2.fit(xt2_train,yt2_train)
t2_pred = d_tree2.predict(xt2_test)
rate_accuracy=accuracy_score(yt2_test,t2_pred)
print(d_tree2.score(xt2_test, yt2_test))
print("Accuracy Rate")
print(rate_accuracy)
plot_tree(d_tree2)


# In[60]:


#Tree3
from sklearn import preprocessing

x3=df[['Perimeter','MajorAxisLength','MinorAxisLength','EquivDiameter','ShapeFactor1']]
y3=df['Class']
xt3_train, xt3_test, yt3_train,yt3_test=train_test_split(x3,y3,test_size=.60, random_state=101)
d_tree3 = DecisionTreeClassifier(max_depth=2, random_state=42)
d_tree3.fit(xt3_train,yt3_train)
t3_pred = d_tree3.predict(xt3_test)
rate_accuracy=accuracy_score(yt3_test,t3_pred)
print(d_tree2.score(xt3_test, yt3_test))
print("Accuracy Rate")
print(rate_accuracy)
plot_tree(d_tree3)


# In[63]:


#knn k nearest
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))


# In[62]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test, y_test))


# In[64]:


print(model.score(x_test, y_test))


# In[65]:


print(model.score(x_test, y_test))


# Logistic regression
# 
# 

# In[67]:


LogisticRegressionModel = LogisticRegression(C=1, solver='newton-cg', class_weight='balanced', multi_class='multinomial',
                            fit_intercept=True, max_iter=100, random_state=44)
LogisticRegressionModel.fit(x_train, y_train)

print(LogisticRegressionModel.score(x_test, y_test))
print(LogisticRegressionModel.score(x_test, y_test))


# In[68]:


y_pred_LR = LogisticRegressionModel.predict(x_test)
CM_LR = confusion_matrix(y_test, y_pred_LR)
sns.heatmap(CM_LR, center=True)
plt.show()
print('Confusion Matrix is\n', CM_LR)


# mean square error for logistic model
# 
# 

# In[69]:


mean_squared_error(y_test,y_pred_LR)


# In[70]:


Linear_model = LinearRegression().fit(x_train,y_train)
Linear_model.score(x_test,y_test)


# In[71]:


Y_pred_L =Linear_model.predict(x_test)
print(Y_pred_L)


# In[72]:


y_test.head()


# In[73]:


Y_pred_L[0:5]


# In[74]:


mean_squared_error(y_test,Y_pred_L)


# Here we have implemented the linear regression and logistic regression on our dataset and find out the mean squared error(MSE) for both of the classifications which is 0.60 and 1.41 resp.

# In[75]:


from sklearn.linear_model import Perceptron
per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(x_train, y_train) 
y_pred = per_clf.predict([[28395,610.291,208.178117,173.888747,1.197191,0.549812,28715,190.141097,0.73923,0.988856,0.958027,0.913358,0.007332,0.003147,0.834222,0.998724]])
print(y_pred)


# In[76]:


import tensorflow as tf
from tensorflow import keras


# In[77]:


tf.__version__


# In[78]:


keras.__version__


# In[79]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[80]:



X_train_full.shape


# In[81]:



X_train_full.dtype


# In[82]:


X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.


# In[83]:


plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()


# In[84]:


y_train


# In[85]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# In[86]:


class_names[y_train[0]]


# In[87]:


X_valid.shape


# In[ ]:


X_test.shape

