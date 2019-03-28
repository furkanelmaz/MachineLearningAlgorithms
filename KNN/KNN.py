import numpy as np
import pandas as pd

class KNN:
    
    def __init__(self,k,distancemetric):
        
        # Initialization
        # k is the number of neighbors
        # distance metric can be 'manhattan' or 'euclidian' (Can be improved)
        # type can be 'regression' or 'classification' (currently only classification)
        self.k = k
        self.dm = distancemetric
        #self.type = type
    
    def train(self,X_train,y_train):
        
        # Not a real training, only importing the training set
        self.X_train = X_train
        self.y_train = y_train
    
    def __distance(self,X_test):
        
        # Distance calculator depends on the distance metric
        # This function is called for one row of X_test, this row is substracted from each row of X_train
        # then summed along columns to determine distances between each point of X_train and corresponding
        # X_test row
        
        if self.dm == 'manhattan':
            
            return np.sum( abs( self.X_train - X_test) , axis = 1)
        
        elif self.dm == 'euclidean':
            
            return np.sum( np.sqrt( (self.X_train - X_test) **2 ), axis = 1)
    
    def predict(self,X_test):
    
        predictions = []
        
        for i in range(0,X_test.shape[0]):         
        
            #Calculate all distances for ith sample of test set
            distance = self.__distance(X_test.iloc[i,:]) 
            
            #Find indices of k number of minimum distances
            neighbors = np.argsort(distance)[0:self.k] 
            
            #Find output values of the found indices in the training set and calculate mode (majority vote)
            prediction = self.y_train.iloc[neighbors].mode().iloc[0]
            
            predictions.append(prediction)
        
        return predictions
