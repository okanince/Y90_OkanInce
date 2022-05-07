#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 13:55:02 2022

@author: okanreyiz
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import recall_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

veriler = pd.read_excel("korelasyon yapıldı.xlsx")
clin = pd.read_excel("Clinical Feats Ready.xlsx")
x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1]
#%% Time
from time import time

#%% Preprocessing

from sklearn.preprocessing import StandardScaler, RobustScaler
rb = RobustScaler()
sc = StandardScaler()
X = sc.fit_transform(x)
X = pd.DataFrame(X, columns = x.columns)

clin_num = clin.drop(["sex 1=m 0=f","Thera or SIR-Spheres thera0"]
                     , axis = 1,inplace = False)
sc1 = StandardScaler()
X_clin = sc1.fit_transform(clin_num)
X_clin = pd.DataFrame(X_clin, columns = clin_num.columns)
X_clin["Sex"] = clin["sex 1=m 0=f"]
X_clin["Spheres"] = clin["Thera or SIR-Spheres thera0"]
X = pd.concat([X,X_clin], axis = 1)

#%%
X.drop("Label", axis = 1, inplace = True)

#%% SMOTE

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
smote = SMOTE(sampling_strategy= .55, random_state=0)
under = RandomUnderSampler(.7, random_state=0)
X_sm,y_sm = smote.fit_resample(X, y)
X_sm, y_sm = under.fit_resample(X_sm, y_sm)






#%%  Wrapper-based feature selection (StandardScaled )

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(C = 1, penalty = "l2", solver = "liblinear",
                          random_state = 42, max_iter = 1000, n_jobs = -1)

sfs1 = SFS(logr, k_features = "best", forward = True, floating = True,
           cv = 5, scoring = "roc_auc", n_jobs = -1, verbose = 2)


sfs1.fit(X_sm,y_sm)



sfs3 = SFS(logr, k_features = "best", forward = False, floating = False,
           cv = 5, scoring = "roc_auc", n_jobs = -1, verbose = 2)


sfs3.fit(X_sm,y_sm)



#%% Wrapper Transform

X1 = sfs1.transform(X_sm)  # Forward prop
X3 = sfs3.transform(X_sm)  # Backward prop


X1 = pd.DataFrame(X1, columns = sfs1.k_feature_names_)
X3 = pd.DataFrame(X3, columns = sfs3.k_feature_names_)


#%% SVM w/CV
from sklearn.metrics import classification_report , confusion_matrix, roc_auc_score 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

specificity = make_scorer(recall_score, pos_label = 0)

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state= 0)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state= 0)

params = {"C": [i for i in (np.arange(.2,13,0.1))] }
svm_cv = SVC(C =1.3, kernel  ="linear", gamma = "scale", shrinking = True,
         max_iter = -1, random_state = 42 )

gs = GridSearchCV(svm_cv, param_grid = params, scoring = "accuracy",
                  n_jobs =-1, cv  =inner, verbose = 1, refit = True)

cs1 = cross_validate(gs, X3, y = y_sm, scoring = ["accuracy", "roc_auc",
                                                "precision","recall","f1"],
                      cv = outer, n_jobs=-1, verbose = 1, return_train_score= True,
                      return_estimator= True)

spec1 = cross_val_score(gs, X3, y=y_sm, scoring = specificity, cv = outer, n_jobs = -1)

#%% Logistic Regression w/CV
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

log2 = LogisticRegression(penalty= "l2",C = 25, solver = "liblinear", 
                          random_state=42, max_iter= 1500, n_jobs=-1)

params_lr = {"C": [i for i in (np.arange(1,30,1))],
             "solver": [ "liblinear","sag", "saga", "lbfgs"]}

gs2 = GridSearchCV(log2, param_grid = params_lr, scoring = "accuracy",
                  n_jobs =-1, cv  =inner, verbose = 1, refit = True)


cs2 = cross_validate(gs2, X3, y = y_sm, scoring = ["accuracy", "roc_auc",
                                                "precision","recall","f1"],
                      cv = outer, n_jobs=-1, verbose = 1, return_train_score= True,
                      return_estimator= True)

spec2 = cross_val_score(gs2, X3, y=y_sm, scoring = specificity, cv = outer, n_jobs = -1)
#%% kNN w/CV
outer = StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights= "distance", algorithm= "auto",
                           metric= "minkowski",p =3, n_jobs = -1,
                           leaf_size=3)



params_knn = {"n_neighbors": [i for i in range(2,7)],
             "weights": [ "uniform", "distance"],
             "algorithm": ["auto", "kd_tree", "ball_tree", "brute"],
             "p": [1,2,3,4,5],
             "leaf_size": [j for j in range(1,17)]}

gs3 = GridSearchCV(knn, param_grid = params_knn, scoring = "accuracy",
                  n_jobs =-1, cv  =inner, verbose = 1, refit = True)

cs3 = cross_validate(gs3, X3, y = y_sm, scoring = ["accuracy", "roc_auc",
                                                "precision","recall","f1"],
                      cv = outer, n_jobs=-1, verbose = 1, return_train_score= True,
                      return_estimator= True)

spec3 = cross_val_score(gs3, X3, y=y_sm, scoring = specificity, cv = outer, n_jobs = -1)



#%% Random Forest w/CV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

outer = StratifiedKFold(n_splits=5, shuffle=True ,random_state=42)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state =42)


rf  = RandomForestClassifier(n_estimators= 100 , criterion="gini",
                             max_depth=4, min_samples_split=2, min_samples_leaf=1,
                             max_features="auto", n_jobs=-1,
                             verbose = 1, max_samples=.5)

params_rf = {"criterion": ["gini", "entropy"],
             "max_depth": [i for i in range(4,10)],
             "min_samples_split": [2,3],
             "min_samples_leaf": [1,2,3],
             "max_samples": [i for i in (np.arange(.3,1,.1))],
             "max_leaf_nodes": [i for i in range(3,30)]}


gs4 = RandomizedSearchCV(estimator= rf, param_distributions= params_rf,
                        scoring = "accuracy", n_iter = 60,
                        n_jobs = -1, refit = True, random_state=42,
                        cv = inner, verbose = 1)

cs4 = cross_validate(gs4, X3, y = y_sm, scoring = ["accuracy", "roc_auc",
                                                "precision","recall","f1"],
                      cv = outer, n_jobs=-1, verbose = 1, return_train_score= True,
                      return_estimator= True)
spec4 = cross_val_score(gs4, X3, y=y_sm, scoring = specificity, cv = outer,
                        n_jobs = -1)
#%%LightGBM
from lightgbm import LGBMClassifier

outer = StratifiedKFold(n_splits=5, shuffle=True ,random_state=0)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state =0)

lgb = LGBMClassifier(boosting_type = "gbdt", num_leaves = 10, max_depth = 7,
                     learning_rate = 0.15, n_estimators = 250,
                     objective = "binary", min_child_samples = 2,
                     subsample = .9,subsample_freq = 5, 
                     colsample_bytree = 1, colsample_bynode = .5,
                     is_unbalance = True, reg_alpha = .6, reg_lambda = 0,
                     random_state = 0, n_jobs = -1,metric = "auc")


params_lgb = {"max_depth": [7,8],
              "num_leaves": [10,11,12,13,14],
              "subsample": [.6,.7,.8,1],
              "subsample_freq": [5,10,15,20]}

gs5 = GridSearchCV(lgb, param_grid = params_lgb, scoring = "accuracy",
                  n_jobs =-1, cv  =inner, verbose = 1, refit = True)



cs5 = cross_validate(gs5, X3, y = y_sm, scoring = ["accuracy", "roc_auc",
                                                "precision","recall","f1"],
                      cv = outer, n_jobs=-1, verbose = 1, return_train_score= True,
                      return_estimator= True)

spec5 = cross_val_score(gs5, X3, y=y_sm, scoring = specificity, cv = outer, n_jobs = -1)



#%% CV Results Gathering
c1 = pd.DataFrame.from_dict(cs1, orient = "index")
c2 = pd.DataFrame.from_dict(cs2, orient = "index")
c3 = pd.DataFrame.from_dict(cs3, orient = "index")
c4 = pd.DataFrame.from_dict(cs4, orient = "index")
c5 = pd.DataFrame.from_dict(cs5, orient = "index")

spec1 = pd.DataFrame(spec1, columns = ["Specificity"])
spec2 = pd.DataFrame(spec2, columns = ["Specificity"])
spec3 = pd.DataFrame(spec3, columns = ["Specificity"])
spec4 = pd.DataFrame(spec4, columns = ["Specificity"])
spec5 = pd.DataFrame(spec5, columns = ["Specificity"])

c1 = pd.concat([c1,spec1.T], axis = 0)
c2 = pd.concat([c2,spec2.T], axis = 0)
c3 = pd.concat([c3,spec3.T], axis = 0)
c4 = pd.concat([c4,spec4.T], axis = 0)
c5 = pd.concat([c5,spec5.T], axis = 0)


#%%
c1.to_excel("SVM.xlsx")
c2.to_excel("logr.xlsx")
c3.to_excel("kNN.xlsx")
c4.to_excel("RF.xlsx")
c5.to_excel("LGB.xlsx")




#%%

gs.fit(X3,y_sm)
gs2.fit(X3,y_sm)
gs4.fit(X3,y_sm)
gs5.fit(X3,y_sm)


#%% ROC visuals - SVM
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc

cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)

clf = gs.best_estimator_

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

fig, ax = plt.subplots(figsize = (10,8))


X4 = X3.values

for i, (train,test) in enumerate(cv.split(X4,y_sm)):
    clf.fit(X4[train],y_sm[train])
    viz = RocCurveDisplay.from_estimator(clf,
                                         X4[test],
                                         y_sm[test],
                                         name = "ROC fold {}".format(i+1),
                                         alpha = 0.3,
                                         lw = 1,
                                         ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0,1], [0,1], linestyle = "--", lw = 2, color = "Black", 
        label = "Chance", alpha = .8)

mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
        mean_fpr,
        mean_tpr,
        color = "b",
        label= r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw = 2,
        alpha = .8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 SD",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    ylabel = "Sensitivity",
    xlabel = "1-Specificity"
)
ax.set_title("Support Vector Machine\n", fontsize = 14)
ax.legend(loc="lower right", fontsize = 9)
plt.savefig("SVM Roc.png", dpi = 300, format = "png")
plt.show()

#%% LR -viz

cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)

clf = gs2.best_estimator_


tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

fig, ax = plt.subplots(figsize = (10,8))


X4 = X3.values

for i, (train,test) in enumerate(cv.split(X4,y_sm)):
    clf.fit(X4[train],y_sm[train])
    viz = RocCurveDisplay.from_estimator(clf,
                                         X4[test],
                                         y_sm[test],
                                         name = "ROC fold {}".format(i+1),
                                         alpha = 0.3,
                                         lw = 1,
                                         ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0,1], [0,1], linestyle = "--", lw = 2, color = "Black", 
        label = "Chance", alpha = .8)

mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
        mean_fpr,
        mean_tpr,
        color = "b",
        label= r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw = 2,
        alpha = .8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 SD",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    ylabel = "Sensitivity",
    xlabel = "1-Specificity"
)
ax.set_title("Logistic Regression\n", fontsize = 14)
ax.legend(loc="lower right", fontsize = 9)
plt.savefig("LR Roc.png", dpi = 300, format = "png")
plt.show()

#%% RF - viz

cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)

clf = gs4.best_estimator_


tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

fig, ax = plt.subplots(figsize = (10,8))


X4 = X3.values

for i, (train,test) in enumerate(cv.split(X4,y_sm)):
    clf.fit(X4[train],y_sm[train])
    viz = RocCurveDisplay.from_estimator(clf,
                                         X4[test],
                                         y_sm[test],
                                         name = "ROC fold {}".format(i+1),
                                         alpha = 0.3,
                                         lw = 1,
                                         ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0,1], [0,1], linestyle = "--", lw = 2, color = "Black", 
        label = "Chance", alpha = .8)

mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
        mean_fpr,
        mean_tpr,
        color = "b",
        label= r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw = 2,
        alpha = .8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 SD",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    ylabel = "Sensitivity",
    xlabel = "1-Specificity"
)
ax.set_title("Random Forest\n", fontsize = 14)
ax.legend(loc="lower right", fontsize = 9)
plt.savefig("RF Roc.png", dpi = 300, format = "png")
plt.show()

#%% LGB- viz

cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0)

clf = gs5.best_estimator_

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

fig, ax = plt.subplots(figsize = (10,8))


X4 = X3.values

for i, (train,test) in enumerate(cv.split(X4,y_sm)):
    clf.fit(X4[train],y_sm[train])
    viz = RocCurveDisplay.from_estimator(clf,
                                         X4[test],
                                         y_sm[test],
                                         name = "ROC fold {}".format(i+1),
                                         alpha = 0.3,
                                         lw = 1,
                                         ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0,1], [0,1], linestyle = "--", lw = 2, color = "Black", 
        label = "Chance", alpha = .8)

mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
        mean_fpr,
        mean_tpr,
        color = "b",
        label= r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw = 2,
        alpha = .8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 SD",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    ylabel = "Sensitivity",
    xlabel = "1-Specificity"
)
ax.set_title("LightGBM\n", fontsize = 14)
ax.legend(loc="lower right", fontsize = 9)
plt.savefig("LightGBM Roc.png", dpi = 300, format = "png")
plt.show()





