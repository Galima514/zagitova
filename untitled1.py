#1
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
from sklearn import linear_model, datasets
import seaborn as sns
from mpl_toolkits import mplot3d
import itertools
from pandas import DataFrame

#2
def LDA_score(X,MU_k,SIGMA,pi_k):
    return (np.log(pi_k) - 1/2 * (MU_k).T @ 
            np.linalg.inv(SIGMA)@(MU_k) +
            X.T @ np.linalg.inv(SIGMA)@ (MU_k)).flatten()[0]
def predict_LDA_class(X,MU_list, SIGMA,pi_list):
    scores_list = []
    classes = len(MU_list)
    
    for p in range(classes):
        score = LDA_score(X.reshape(-1,1),MU_list[p] 
                          .reshape(-1,1), SIGMA,pi_list[0])
        scores_list.append(score)
   
    return np.argmax(scores_list)

#3
def QDA_score(X,MU_k,SIGMA,pi_k):
    SIGMA_inv = np.linalg.inv(SIGMA)
    return (np.log(pi_k) - 1/2 *
        np.log(np.linalg.det(SIGMA_inv)) - 1/2 *
        (X - MU_k).T @ SIGMA_inv @ (X - MU_k)).flatten()[0]
def predict_QDA_class(X,MU_list, SIGMA_list, pi_list):
    scores_list = []
    classes = len(MU_list)
    
    for p in range(classes):
        score = QDA_score(X.reshape(-1,1),MU_list[p]
                          .reshape(-1,1),SIGMA_list[p],pi_list[p])
        scores_list.append(score)
    return np.argmax(scores_list)

#4
def lambdaSW(data, size_l, size_r):
    s = []
    for i in data.iloc[:,size_l:size_r]:
        s.append(round(shapiro(data[i])[0],5))
    return s

#5
def Accuracy(X, y):
    tree = DecisionTreeClassifier(random_state=0)
    scores_1 = bootstrap_point632_score(tree, X, y, method='.632')
    scores_2 = bootstrap_point632_score(tree, X, y, method='.632+')
    acc1 = np.mean(scores_1)
    acc2 = np.mean(scores_2)
    return [round(acc1, 4), round(acc2, 4)]

#7
pharmacy = sns.load_dataset("pharmacy")
pharmacy = pharmacy.rename(columns={'species': 'category_name'}) 
pharmacy['category_int'] = pharmacy.category_name.astype('category').cat.codes
pharmacy_train, pharmacy_test = train_test_split(pharmacy, test_size=0.2, random_state=42) 
class_count = pharmacy_train['category_name'].nunique()
class_unique_count = pharmacy_train['category_name'].value_counts().values
class_all_count = pharmacy_train['category_name'].count()
 class_name =pharmacy_train['category_name'].unique()
X_data =pharmacy_test.iloc[:,0:-2]
y_labels = pharmacy_test['category_int'].copy()
y_pred = list()
colors=["#000099","#009900", "#449999", "#994444"]

#8
species_g = pharmacy_train.iloc[:,0:-1].groupby('category_name') 
    .mean().values
split_s_g = [(i+1) for i in range(class_count-1)]
mu_list = np.split(species_g, split_s_g) sigma = iris_train.iloc[:,0:-2]
    .cov().values 
pi_list = class_unique_count / class_all_count

y_pred.append(np.array( [predict_LDA_class( np.array(row) 
                                           .reshape(-1,1), mu_list, sigma, pi_list)
                                             for row in X_data.to_numpy() ] ))
#9
sigma_g = pharmacy_train.iloc[:,0:-1].groupby('category_name').cov() 
sigma_list = np.split(sigma_g.values,[4,8], axis = 0)
y_pred.append(np.array( [predict_QDA_class( row. reshape(-1,1),
                        mu_list, sigma_list, pi_list)
                        for row in X_data.to_numpy() ] ))

#10
classes = np.append(class_name,["error"], axis=0) 
fig = plt.figure(figsize=(9, 18))
current_axis = fig.add_subplot(3, 1, 1)
current_axis.set_title('LDA')
X_show = np.array(X_data.values)
X_show = np.append(X_show, np.array(y_pred[0]).reshape(-1, 1), axis = 1)
show_df = DataFrame(X_show)
show_df.columns = ['0','1','2','3', 'class']
show_df['class'] = show_df['class'].apply(lambda x: classes[int(x)]) 
pd.plotting.andrews_curves(show_df, 'class', ax=current_axis, color=colors)
al = Accuracy(X_show[:,0:-1], X_show[:,-1])
current_axis = fig.add_subplot(3, 1, 2)
current_axis.set_title('QDA')
X_show = np.array(X_data.values) X_show = np.append(X_show, np.array(y_pred[1]).reshape(-1, 1), axis = 1)
show_df = DataFrame(X_show)
show_df.columns = ['0','1','2','3', 'class']
show_df['class'] = show_df['class'].apply(lambda x: classes[int(x)])
pd.plotting.andrews_curves(show_df, 'class', ax=current_axis, color=colors)
aq = Accuracy(X_show[:,0:-1], X_show[:,-1])
current_axis = fig.add_subplot(3, 1, 3)
current_axis.set_title('Target')
pd.plotting.andrews_curves(pharmacy_test.iloc[:,0:-1], 'category_name', ax=current_axis, color=colors)
plt.show()
print(al, aq)

#11


#12


