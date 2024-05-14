import numpy as np
from sklearn.utils import resample
from copy import deepcopy

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class AggregatedCP:
    """ 
    Implementation of Aggregated Conformal Prediction(ACP)
    Related Papers:
    1/ Lars Carlsson, Martin Eklund and Ulf Norinder. Aggregated conformal prediction.
    2/ Vladimir Vovk. Cross-conformal predictors.
    """
    def __init__(self,  B1 = 10, classifier = "RandomForest", alpha = 0.05, mode = "CCP"):
        self.model = classifier
        self.B1 = B1
        self.alpha = alpha
        self.mode = mode
        self.modal = classifier
    
    def split(self, trainset, random_state = 42):
        # Dataset split method for different type 0f ACP
        n_train = len(trainset[1])
        n_class = len(set(trainset[1]))
        dataset_pair = []

        if self.mode == "ICP":
            idx_grouped = [np.where(trainset[1] == i)[0] for i in range(n_class)]
            for i in range(n_class):
                np.random.shuffle(idx_grouped[i])
            splits = [int(0.7 * len(idx_grouped[i])) for i in range(n_class)]

            train_icp = np.concatenate([trainset[0][idx_grouped[i][:splits[i]]] for i in range(n_class)]), \
                        np.concatenate([trainset[1][idx_grouped[i][:splits[i]]] for i in range(n_class)])
            calib_icp = np.concatenate([trainset[0][idx_grouped[i][splits[i]:]] for i in range(n_class)]), \
                        np.concatenate([trainset[1][idx_grouped[i][splits[i]:]] for i in range(n_class)])

            dataset_pair.append((train_icp, calib_icp))

        elif self.mode == "CCP":
            idx_grouped = [np.where(trainset[1] == i)[0] for i in range(n_class)]
            for i in range(n_class):
                np.random.shuffle(idx_grouped[i])

            ntrain_per_class = int(n_train/n_class)
            cross_fold = 2 if ntrain_per_class < 500 else 5 if 500 < ntrain_per_class < 1000 else 10
            print("CCP with", cross_fold, "folds")
            nfold_per_class = int(ntrain_per_class/cross_fold)
            for b1 in range(cross_fold):
                if b1 != cross_fold-1:
                    train_b1 = np.concatenate([trainset[0][idx_grouped[i][nfold_per_class*b1:nfold_per_class*(b1+1)]] for i in range(n_class)]), \
                               np.concatenate([trainset[1][idx_grouped[i][nfold_per_class*b1:nfold_per_class*(b1+1)]] for i in range(n_class)])
                    calib_b1 = np.concatenate([np.delete(trainset[0][idx_grouped[i]], range(nfold_per_class*b1, nfold_per_class*(b1+1)), axis=0) \
                                              for i in range(n_class)]), \
                               np.concatenate([np.delete(trainset[1][idx_grouped[i]], range(nfold_per_class*b1, nfold_per_class*(b1+1)), axis=0) \
                                              for i in range(n_class)])
                else:
                    train_b1 = np.concatenate([trainset[0][idx_grouped[i][nfold_per_class*b1:]] for i in range(n_class)]), \
                               np.concatenate([trainset[1][idx_grouped[i][nfold_per_class*b1:]] for i in range(n_class)])
                    calib_b1 = np.concatenate([trainset[0][idx_grouped[i][:nfold_per_class*b1]] for i in range(n_class)]), \
                               np.concatenate([trainset[1][idx_grouped[i][:nfold_per_class*b1]] for i in range(n_class)])
                dataset_pair.append((train_b1,calib_b1))
        
        elif self.mode == "BCP":
            for b1 in range(self.B1):
                bootstrap_indices = resample(range(n_train), replace=True, n_samples=int(n_train), random_state=random_state)
                train_b1 = trainset[0][bootstrap_indices], trainset[1][bootstrap_indices]
                calib_b1 = np.delete(trainset[0], bootstrap_indices, axis=0), np.delete(trainset[1], bootstrap_indices, axis=0)
                dataset_pair.append((train_b1, calib_b1))

        self.dataset_sp = dataset_pair

    
    def predict(self, trainset, testset, random_state = 42):
        # Data standardization and trainset split
        scaler = StandardScaler()
        train_scale = scaler.fit_transform(trainset[0])
        test_scale = scaler.transform(testset[0])
        trainset = train_scale, trainset[1]
        testset = test_scale, testset[1]
        self.split(trainset)

        # Perform ACP
        pvals_list = []
        for sp in range(len(self.dataset_sp)):
            if self.model == "RandomForest":
                c_item = RandomForestClassifier(n_estimators=100, random_state = random_state+sp)
            elif self.model == "Logistic":
                c_item = LogisticRegression()
            elif self.model == "LDA":
                c_item = LinearDiscriminantAnalysis()
            elif self.model == "QDA":
                c_item = QuadraticDiscriminantAnalysis()
            elif self.model == "SVM":
                c_item = SVC(probability=True)
            elif self.model == "KNN":
                c_item = KNeighborsClassifier()
                # c_item = KNeighborsClassifier(n_neighbors=100)
            
            c_item.fit(self.dataset_sp[sp][0][0],self.dataset_sp[sp][0][1])
            calib_scores = c_item.predict_proba(self.dataset_sp[sp][1][0])
            test_scores = c_item.predict_proba(testset[0])


            cp = np.zeros_like(test_scores)
            for i in range(len(test_scores)):
                for j in range(calib_scores.shape[1]):
                    cp[i,j] = (1 + np.sum(calib_scores[:, j] < test_scores[i, j]))/(len(calib_scores)+1)
            pvals_list.append(cp)
        average_cp = np.sum(pvals_list, axis=0)/len(self.dataset_sp)
        # print(average_cp.shape)
        # print("average cp:", average_cp[:3], testset[1][:3])

        # Compute metrics: Accuracy, Coverage, Efficiency
        contain = 0
        len_set = 0
        correct = 0
        for i in range(len(test_scores)):
            cp_set = []
            for j in range(test_scores.shape[1]):
                if average_cp[i,j] > np.ceil(((len(calib_scores)+1)*(self.alpha)))/(len(calib_scores)+1):
                    cp_set.append(j)
            if testset[1][i] in cp_set:
                contain += 1
                len_set += len(cp_set)
                if len(cp_set) ==1:
                    correct += 1 
        
        return correct/len(testset[1]), contain/len(testset[1]), len_set/contain
    


                        

    



