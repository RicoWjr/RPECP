import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt

"""
    Some exploratory codes.
"""

def check_features(data):
    mean, var = [],[]
    for p in range(data.shape[1]):
        mean.append(np.mean(data.T[p]))
        var.append(np.var(data.T[p]))
    cov = np.cov(data.T)
    return mean,var,cov


def select_features(data,features):
    """ data: np.array 
        features: list """
    dt = data.T 
    res = []
    for i in features:
        res.append(dt[i])
    return np.array(res).T


def diversity(data_train, data_test, filt_method = "abs_mean_diff" ):
    mean_train, var_train = check_features(data_train)
    mean_test, var_test = check_features(data_test)
    if filt_method == "abs_mean_diff":
        output = select_features(data_test,abs_mean_diff(mean_train, mean_test, dist = 0.1))
    if filt_method == "KL":
        output = select_features(data_test,abs_mean_diff(mean_train, mean_test, dist = 0.1))
"""     if filt_method == "Corrp":
        output =  """

def abs_mean_diff(mean_train, mean_test, dist = 0.1):
    features_selected = []
    for i in range(len(mean_train)):
        if  abs(mean_train[i]-mean_test[i])>dist:
            features_selected.append(i)
    return features_selected

def JS_divergence(data_train, data_test):
    features_selected = []
    data_train_T, data_test_T = data_train.T, data_test.T
    for i in range(len(data_train_T)):
        p,q = data_train_T[i], data_test_T[i]
        M = (p+q)/2
        JS = 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)
        if JS > 0.1:
            features_selected.append(i)

    return features_selected


def Lasso_features(data1,data2,threshold = 0.05):
    lasso = Lasso(alpha=threshold)
    data_l = np.vstack((data1,data2))
    label_l = np.concatenate((np.ones(len(data1)),np.zeros(len(data2))))
    lasso.fit(data_l, label_l)
    fs = []
    for f in range(len(lasso.coef_)):
        if lasso.coef_[f] != 0:
            fs.append(f)
    data1_s, data2_s = [],[]
    for cs in fs:
        data1_s.append(data1.T[cs])
        data2_s.append(data2.T[cs])
    return np.array(data1_s).T,np.array(data2_s).T,np.array(fs)


def feature_plot(data1, data2, f_i):
    plt.rcParams["axes.unicode_minus"]=False 
    fig, ax = plt.subplots(figsize=(12,9))
    ax.hist(data1.T[f_i].T, bins=16, alpha = 0.7, density=True, label="inlier")
    ax.hist(data2.T[f_i].T, bins=16, alpha = 0.7, density=True, label="ouitlier")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    from generate import data

    p = 200

    mean1 = np.repeat(0,p)
    mean0 = np.concatenate((np.repeat(3,100),np.repeat(0,p-100)))
    mean = [mean1, mean0]
    

    cov1 = np.diag(np.repeat(1,p))
    cov0 = np.diag(np.concatenate((np.repeat(3,100),np.repeat(1,p-100))))
    cov = [cov1, cov0]

    X = data(mean, cov, mixture = False)
    X_train = X.generate(2000, purity=1)
    X_test = X.generate(1500, purity=2/3)

    print(X_train[0].shape, X_test[0].shape)
    f = Lasso_features(X_train[0], X_test[0])
    print(len(f),len(f[f<=100]))
    print(f)