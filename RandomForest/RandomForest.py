import random
import pandas as pd
import numpy as np
from MachineLearningAlgorithms.RandomForest.DT_for_RF import DT_RF

class RandomForest:
    
    def __init__(self,min_samples = 2, max_depth = 999, n_tree = 5, n_features=5):
        
        self.n_tree = n_tree # How many trees will be used in the forest
        self.min_samples = min_samples # Same as original Decision Tree
        self.max_depth = max_depth # Same as original Decision Tree
        self.counter = 0 # Same as original Decision Tree
        self.best_entropy = 10 # Same as original Decision Tree
        self.n_features = n_features # Selects how many features will be used during growing
                                     # the trees
    
    def train(self,df):
        
        m , n = df.shape
        total_list = list(range(0,m))
        
        self.trees = list()
        
        # Create trees in the forest
        for i in range(0,self.n_tree):
            
            # Create a bootstrapped data set from the original one
            bootstrapped_ind = random.choices(total_list, k = len(total_list))
            
            bootstrapped_set = df.iloc[bootstrapped_ind,:].copy()
                        
            # Send bootstrapped data set to the original decision tree algorithm with twist
            # (check DT_for_RF)
            self.trees.append(DT_RF(self.min_samples,self.max_depth,self.n_features))
            
            self.trees[i].train_a_tree(bootstrapped_set)
    
    # Predict the output of the new data using the growed trees, use majority vote to decide
    # final outcome
    def predict(self,df):
        
        self.predictions = list()
        
        for i in range(0,df.shape[0]):
        
            prediction = list()
            
            for j in range(0,self.n_tree):
                
                prediction.append(self.trees[j]._predict_for_one(df.iloc[i,:]))
                
            self.predictions.append(max(set(prediction), key=prediction.count))
        
        return self.predictions
            
            

      