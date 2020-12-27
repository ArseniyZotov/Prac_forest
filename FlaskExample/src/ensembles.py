import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error

class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        
        self.n_estimators_ = n_estimators
        self.max_depth_ = max_depth
        if feature_subsample_size is None:
            self.feature_subsample_size_ = 1./3
        else:
            self.feature_subsample_size_ = feature_subsample_size
        
        self.trees_parameters_ = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features
            
        y_val : numpy ndarray
            Array of size n_val_objects           
       
        if X_val != None returns
        -------
        rmse : numpy ndarray
            Array of size n_estimators
        """
        
        n_features = int(self.feature_subsample_size_ * X.shape[1])
        self.trees_ = [DecisionTreeRegressor(max_depth=self.max_depth_, **self.trees_parameters_) 
                          for i in range(self.n_estimators_)]
        self.indices_ = np.zeros((self.n_estimators_, n_features), dtype=int)
        rmse = np.zeros(self.n_estimators_)
        val_pred_sum = 0
        
        for i in range(self.n_estimators_):
            self.indices_[i] = np.random.choice(X.shape[1], n_features, replace=False)
            
            self.trees_[i].fit(X[:, self.indices_[i]], y)
            
            if X_val is not None:
                val_pred_sum += self.trees_[i].predict(X_val[:, self.indices_[i]])
                rmse[i] = mean_squared_error(val_pred_sum/(i+1), y_val, squared=False)
        
        if X_val is not None:
            return rmse
        
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        
        answer = np.zeros(X.shape[0])
            
        for i in range(self.n_estimators_):
            answer += self.trees_[i].predict(X[:, self.indices_[i]])
        
        return answer/self.n_estimators_
    
class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators_ = n_estimators
        self.max_depth_ = max_depth
        self.learning_rate_ = learning_rate
        
        if feature_subsample_size is None:
            self.feature_subsample_size_ = 1./3
        else:
            self.feature_subsample_size_ = feature_subsample_size
        
        self.trees_parameters_ = trees_parameters
        
    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
            
        X_val : numpy ndarray
            Array of size k_objects, n_features
            
        y_val : numpy ndarray
            Array of size k_objects
            
        if X_val != None returns
        -------
        rmse : numpy ndarray
            Array of size n_estimators
        """
        
        pred_y = np.zeros(X.shape[0])
        
        n_features = int(self.feature_subsample_size_ * X.shape[1])
        self.coefs_ = np.zeros(self.n_estimators_)
        self.trees_ = [DecisionTreeRegressor(max_depth=self.max_depth_, **self.trees_parameters_) 
                          for i in range(self.n_estimators_)]
        self.indices_ = np.zeros((self.n_estimators_, n_features), dtype=int)
        rmse = np.zeros(self.n_estimators_)
        val_pred_sum = 0
        
        for i in range(self.n_estimators_):
            grad = y - pred_y
            self.indices_[i] = np.random.choice(X.shape[1], n_features, replace=False)
            
            X_temp = X[:, self.indices_[i]]
            self.trees_[i].fit(X_temp, grad)
            
            tree_pred = self.trees_[i].predict(X_temp)
            self.coefs_[i] = minimize_scalar(lambda x: ((grad - x*tree_pred)**2).sum()).x
            pred_y += self.learning_rate_ * self.coefs_[i] * tree_pred
            
            if X_val is not None:
                val_pred_sum += self.trees_[i].predict(X_val[:, self.indices_[i]]) * self.coefs_[i]
                rmse[i] = mean_squared_error(val_pred_sum*self.learning_rate_, y_val, squared=False)
        
        if X_val is not None:
            return rmse
            
    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        answer = np.zeros(X.shape[0])
            
        for i in range(self.n_estimators_):
            answer += self.trees_[i].predict(X[:, self.indices_[i]]) * self.coefs_[i]
            
        answer *= self.learning_rate_
        
        return answer
