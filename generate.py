import numpy as np
import pandas as pd

class data:
    def __init__(self,mean,cov,mixture = False):
        self.mix = mixture
        self.data_store = []
        self.mean = mean
        self.cov = cov
        self.z = np.random.uniform(low=-3, high=3, size=(len(self.mean[0]), len(self.mean[0])))
        # self.z = np.random.uniform(3, 1, size=(len(self.mean[0]), len(self.mean[0])))
    
    def generate(self, n, purity, pow = 1, shuffle = False, random_state = 42):
        np.random.seed(random_state)
        inliers = len(self.mean)-1
        n_inlier = int(n*purity/inliers)
        n_outlier = n-n_inlier*inliers
        data, idx = None, None
        for i in range(inliers):
            data_in = np.random.multivariate_normal(self.mean[i], self.cov[i], n_inlier, check_valid="ignore")
            if pow > 1:
                data_in = np.float_power(pow,data_in)
            idx_in = np.repeat(i+1, n*purity/inliers)
            if i == 0:
                data, idx = data_in, idx_in
            else:
                data = np.vstack((data,data_in))
                idx = np.concatenate((idx,idx_in))
        if(purity<1):
            data_out = np.random.multivariate_normal(self.mean[-1], self.cov[-1], n_outlier, check_valid="ignore")
            if pow > 1:
                data_out = np.float_power(pow,data_out)
            data = np.vstack((data_out, data))
            idx = np.concatenate((np.repeat(0, n_outlier),idx))
        
        if self.mix:
            cluster_idx = np.random.choice(self.z.shape[0], n, replace=True)
            data = data + self.z[cluster_idx,]
        
        if shuffle == True:
            shuffle_ix = np.random.permutation(np.arange(len(idx)))
            data, idx = data[shuffle_ix], idx[shuffle_ix]
        self.data_store.append((data,idx,purity))
        return data,idx

    def stat(self):
        class_num = len(self.mean)-1
        print("class num:", class_num, "| oneclass outlier detection" if class_num == 1 
              else "| multi-class prediction and outlier detection")
        print("feature num:", len(self.mean[0]))
        print("========================================")
        for i in range(len(self.data_store)):
            dataset = self.data_store[i]
            if(dataset[2] == 1):
                print("training set", i, ":")
                print("    data num:", len(dataset[0]))
                for k in range(class_num):
                    print("        class", k+1, "mean: %.3f" % np.mean(dataset[0][dataset[1]==k+1]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==k+1]))
                print("========================================")
            else:
                print("test set", i, ": ")
                print("    data num:", len(dataset[0]))
                print("    inliers per class:", len(dataset[0][dataset[1]==1]))
                for k in range(class_num):
                    print("        class", k+1, "mean: %.3f" % np.mean(dataset[0][dataset[1]==k+1]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==k+1]))
                print("    outliers:", len(dataset[0][dataset[1]==0]))
                print("        outlier", "mean: %.3f" % np.mean(dataset[0][dataset[1]==0]), 
                          "var: %.3f" % np.var(dataset[0][dataset[1]==0]))
                print("========================================")


if __name__ == "__main__":
    from unused_featureSelect import check_features
    p = 200
    mean1 = np.repeat(0,p)
    mean0 = np.repeat(10,p)
    mean = [mean1,mean0]

    cov1 = np.diag(np.repeat(1,p))
    cov0 = np.diag(np.repeat(3,p))
    cov = [cov1,cov0]

    X = data(mean,cov)
    X_train = X.generate(4000,purity=1)

    train1 = X_train[0][:2000]
    train2 = X_train[0][2000:4000]

    mean_1, var_1, cov_1 = check_features(train1)
    mean_new,cov_new = [mean_1,mean0], [cov_1, cov0]

    X_new = data(mean_new, cov_new)
    X_train_new = X_new.generate(1000,purity=1)

    """ for i in range(5):
        r = np.corrcoef(train2.T[i], train1.T[i])[0][1]
        print(r)

    count_all = []
    for i in range(20):
        min = np.min(train1.T[i])
        max = np.max(train1.T[i])
        mean = np.mean(train1.T[i])
        var = np.var(train1.T[i])
        print("mean:", mean,np.mean(X_new [0].T[i]))
        print("var:", var,np.var(X_new [0].T[i]))
        print("-------------------------")

        count = 0
        for j in range(1000):
            if X_train[0][j][i] >= min and X_train[0][j][i] <= max:
                count += 1
        count_all.append(count)
    print(count_all) """

    from scipy.stats import kstest
    from scipy.stats import shapiro
    from scipy.stats import skew
    res1,res_new = [],[]
    for i in range(p):
        # res_1 = kstest(train2.T[i],cdf="norm")
        # res_n = kstest(X_train_new[0].T[i],cdf="norm")
        # res_1 = shapiro(train2.T[i])
        # res_n = shapiro(X_train_new[0].T[i])
        res_1 = skew(train2.T[i])
        res_n = skew(X_train_new[0].T[i])
        res1.append(res_1)
        res_new.append(res_n)

    # print(res_1)
    # print(res_new)

    c1, c_new = 0,0
    for i in range(len(res1)):
        # if res1[i][1]<0.05:
        if np.abs(res1[i])<0.1:
            c1+=1
    for i in range(len(res_new)):
        # if res_new[i][1]<0.05:
        if np.abs(res1[i])<0.1:
            c_new+=1
    print(c1,c_new)
    