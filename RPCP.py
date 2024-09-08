import random
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class RPEnsemble_CP:
    """
    Our paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4794962
    Xiaoyu Qian, Jinru Wu, Youwu Lin. Random Projection Ensemble Conformal Prediction for High-Dimensional Classification.
    """
    def __init__(self, rp, d = 1, B1 = 10, B2 = 50, classifier = "RandomForest", alpha = 0.05):
        self.rp = rp
        self.model = classifier
        self.d = d
        self.B1 = B1
        self.B2 = B2
        self.alpha = alpha

    def calculate_cp(self, calib_scores, idx_calib):
        t = np.zeros_like(calib_scores[0])
        K = len(t)
        # a = np.floor(self.alpha*(len(calib_scores)+1))/(len(calib_scores+1))
        for k in range(K):
            sorted_probabilities = np.sort(calib_scores[idx_calib==k][:,k])
            t[k] = sorted_probabilities[int(self.alpha * sorted_probabilities.shape[0])]
        return t
    
    def calculate_cp_plsr(self, calib_scores, idx_calib):
        t = np.zeros_like(calib_scores[0])
        K = len(t)
        for k in range(K):
            sorted_probabilities = np.sort(-(calib_scores[idx_calib==k][:,k]-k)**2)
            t[k] = sorted_probabilities[int(self.alpha * sorted_probabilities.shape[0])]
        return t


    def select(self, trainset, random_state):
        """
        Select optimal random projection matrix and classifier
        """
        n_calib = int(len(trainset[0]) * 0.5)
        selected_indices = np.sort(random.sample(range(len(trainset[0])), n_calib))
        data_train = np.array([trainset[0][i] for i in selected_indices])
        idx_train = np.array([trainset[1][i] for i in selected_indices])
        data_calib = np.array([trainset[0][i] for i in range(len(trainset[0])) if i not in selected_indices])
        idx_calib = np.array([trainset[1][i] for i in range(len(trainset[1])) if i not in selected_indices])

        rp_matrix, rp_calib_log, rp_calib_acc = [], [], []
        models = []
        for i in range(self.B2):
            if self.rp == "Gaussian":
                rp_item = GaussianRandomProjection(n_components = self.d, random_state = random_state + i)
            elif self.rp == "axis":
                rp_item = SparseRandomProjection(n_components = self.d, random_state = random_state + i)
            else:
                raise NotImplementedError
            
            if self.model == "RandomForest":
                c_item = RandomForestClassifier(n_estimators=100, random_state = random_state+i)
            elif self.model == "Logistic":
                c_item = LogisticRegression(random_state = random_state+i)
            elif self.model == "LDA":
                c_item = LinearDiscriminantAnalysis()
            elif self.model == "QDA":
                c_item = QuadraticDiscriminantAnalysis()
            elif self.model == "SVM":
                c_item = SVC(probability=True, random_state = random_state+i)
            elif self.model == "KNN":
                c_item = KNeighborsClassifier()
                # c_item = KNeighborsClassifier(n_neighbors=100)
            elif self.model == "PLSDA":
                c_item = PLSRegression()

            if isinstance(c_item, PLSRegression):
                train_rp, calib_rp = rp_item.fit_transform(data_train), rp_item.fit_transform(data_calib)
                idx_train = pd.get_dummies(idx_train)
                c_item.fit(train_rp, idx_train)
                rp_res = c_item.predict(calib_rp)
                rp_logits = rp_res
                rp_res = np.array([np.argmax(i) for i in rp_res])
            else:
                train_rp, calib_rp = rp_item.fit_transform(data_train), rp_item.fit_transform(data_calib)
                c_item.fit(train_rp, idx_train)
                rp_res = c_item.predict(calib_rp)
                rp_logits = c_item.predict_proba(calib_rp)

            rp_matrix.append(rp_item)
            rp_calib_log.append(rp_logits)
            rp_calib_acc.append(accuracy_score(idx_calib, rp_res))
            models.append(c_item)
        
        # choose items which maximize the calibration Accuracy
        self.opt_rp, self.c, calib_scores, _  = max(list(zip(rp_matrix, models, rp_calib_log, rp_calib_acc)), key=lambda x: x[3])
        # print("The max-accuracy is:", _)
        
        if isinstance(c_item, PLSRegression):
            self.cp_tau = self.calculate_cp_plsr(calib_scores,idx_calib)
            self.cp_tau = self.calculate_cp(calib_scores,idx_calib)
        else:
            self.cp_tau = self.calculate_cp(calib_scores,idx_calib)

    def RP_Conformal(self, trainset, testset):
        """ 
        Perform RPEnsemble Conformal Prediction(RPECP)
        """

        # data standardization
        scaler = StandardScaler()
        train_scale = scaler.fit_transform(trainset[0])
        test_scale = scaler.transform(testset[0])
        trainset = train_scale, trainset[1]
        testset = test_scale, testset[1]

        total_sets = []
        for i in range(self.B1):
            self.select(trainset, random_state = 2024*i)
            test_rp = self.opt_rp.fit_transform(testset[0])
            if isinstance(self.c, PLSRegression):
                pvals = self.c.predict(test_rp)
            else:
                pvals = self.c.predict_proba(test_rp)
            # print(self.cp_tau, pvals.shape)
            indices = []
            for j in range(len(testset[0])):
                pred_labels = []
                for k in range(len(self.cp_tau)):
                    # if isinstance(self.c, PLSRegression):
                    #     pvals[j,k] = -(pvals[j,k] - k)**2
                    if pvals[j,k] >= self.cp_tau[k]:
                        pred_labels.append(k)
                indices.append(pred_labels)
            total_sets.append(indices)

        sp_sets = [[item[i] for item in total_sets] for i in range(len(testset[0]))]

        correct,contain = 0,0
        res_len = 0
        for b1 in range(len(sp_sets)):
            count_dict = {}
            flat_list = [item for sublist in sp_sets[b1] for item in sublist]
            if len(flat_list)>0:
                count_dict = Counter(flat_list)
                # res = [number for number, count in count_dict.items() if count >= int((1-4*self.alpha)*len(sp_sets[b1]))]
                res = [number for number, count in count_dict.items() if count >= int(0.5*len(sp_sets[b1]))]
                max_count = max(count_dict.values())
                max_numbers = [number for number, count in count_dict.items() if count == max_count]
                if testset[1][b1] in res:
                    contain += 1
                    res_len += len(res)
                if len(max_numbers) == 1:
                    if testset[1][b1] == max_numbers:
                        correct += 1
                else:
                    for sublist in sp_sets[b1]:
                        if len(sublist) == 1:
                            if testset[1][b1] == sublist[0]:
                                correct += 1
                                break
        if contain != 0:
            average_len = res_len/contain
        else:
            average_len = 0

        return correct/len(sp_sets), contain/len(sp_sets), average_len
                