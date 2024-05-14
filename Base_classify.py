import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression

class base:
    """
    Base classifier without Conformal Prediction
    """
    def __init__(self,classifier = "RandomForest", random_state = 2024):
        np.random.seed(random_state)
        if classifier == "RandomForest":
            self.c = RandomForestClassifier(n_estimators=100, random_state = random_state)
        elif classifier == "Logistic":
            self.c = LogisticRegression()
        elif classifier == "SVM":
            self.c = SVC(random_state=random_state)
        elif classifier == "KNN":
            self.c = KNeighborsClassifier()
            # self.c = KNeighborsClassifier(n_neighbors=100)
        elif classifier == "LDA":
            self.c = LinearDiscriminantAnalysis()
        elif classifier == "QDA":
            self.c = QuadraticDiscriminantAnalysis()
        
             
    def predict(self, X_train, X_test):
        self.c.fit(X_train[0],X_train[1])
        test_res = self.c.predict(X_test[0])
        acc = accuracy_score(X_test[1], test_res)
        
        return acc