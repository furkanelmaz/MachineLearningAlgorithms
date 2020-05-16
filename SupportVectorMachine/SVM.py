import pandas as pd
import numpy as np
import cvxopt
import cvxopt.solvers

class SVM:
    
    def __init__(self, C = 1, kernel = 'linear'):
        
        self.kernel = kernel
        self.C = C
    
    def _linear(self, xi, xj):
        
        return np.dot(xi,xj)
    
    def _polynomial(self, xi, xj):
        
        pass
    
    
    # Converts output classes into +1 and -1 or reverse the conversion, 
    # conversion = 1 to encode , 0 for decode 
    def _convert(self,y,conversion):
        
        if conversion == 1:
            
        # Find unique classes in output column
            self.class_list = np.unique(y)
        
        # Encode the found classes into +1 and -1
            one_indices = np.where(y == self.class_list[0])
            minusone_indices = np.where(y == self.class_list[1])
            
            y[one_indices] = 1
            y[minusone_indices] = -1
        
        else:
            
            # Reverse the conversion
            one_indices = np.where(y == 1)
            minusone_indices = np.where(y == -1)
            
            y[one_indices] = self.class_list[0]
            y[minusone_indices] = self.class_list[1]
        
        return y
            
    def train(self,df):
        
        kernel_names = {'linear': self._linear , 'polynomial': self._polynomial}
        # Converting dataframe to numpy array and specify inputs/output
        y = df.iloc[:,-1].copy().values
        
        X = df.iloc[:,:-1].copy().values
        
        # Converting output classes into +1 and -1
        
        y = self._convert(y, conversion = 1)
    
        # Calculate dot products of X's in dual form of SVM (Kernel)
        
        m , n = X.shape
        
        K = np.zeros((m, m))
        
        for i in range(m):
            
            for j in range(m):
                
                K[i,j] = kernel_names[self.kernel](X[i],X[j]) # Kernel   
        
        # Calculate product of y's in dual form
        
        product_y = np.outer(y,y)
        
        # Converting into quadratic form and prepare for cvxopt
        P = cvxopt.matrix(product_y * K)
        q = cvxopt.matrix(np.ones(m) * -1)
        A = cvxopt.matrix(y, (1,m),'d')
        b = cvxopt.matrix(0.0)         
        G = cvxopt.matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        
        cvxopt.solvers.options['show_progress'] = False
        
        # Find the all alpha values
        self.alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        
        # Calculate weight matrix 
        y_t = y.reshape(-1,1)
        
        self.w = ( (y_t * self.alphas).T @ X ).reshape(-1,1)
        
        # Find nonzero alphas to calculate b
        S = (self.alphas > 1e-4).flatten()
        
        # Calculate b
        self.b = y_t[S] - np.dot(X[S], self.w)
        
    def predict(self,testdata):
        
        # If testdata is not given in numpy array convert it 
        try:
            
            testdata = testdata.values
            
        except:
            
            pass
        
        y_pred = []
        
        # Raw prediction calculated from formula
        raw_pred = (testdata @ self.w + self.b[0] )
        
        # Process raw prediction respect to decision rule
        # 1- If the prediction value is greater than zero then it is the class labeled as +1 class
        # 2- Else, it is the class labeled as -1
        for i in raw_pred:
            
            if i >= 0:
                
                y_pred.append(1)
                
            elif i < 0:
                
                y_pred.append(-1)
        
        # Reverse the conversion of the outputs
        y_pred = self._convert(np.array(y_pred), conversion = 0)
        
        return y_pred
