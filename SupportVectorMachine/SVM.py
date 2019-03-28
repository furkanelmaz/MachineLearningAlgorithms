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
    
    def train(self,df):
        
        kernel_names = {'linear': self._linear , 'polynomial': self._polynomial}
        # Converting dataframe to numpy array and specify inputs/output
        y = df.iloc[:,-1].values
        
        X = df.iloc[:,:-1].values
        
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
        
        self.alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        
        y_t = y.reshape(-1,1)
        
        self.w = ( (y_t * self.alphas).T @ X ).reshape(-1,1)
        
        S = (self.alphas > 1e-4).flatten()
        
        self.b = y_t[S] - np.dot(X[S], self.w)
        
    def predict(self,testdata):
        
        try:
            
            testdata = testdata.values
            
        except:
            
            pass
        
        y_pred = []
        raw_pred = (testdata @ self.w + self.b[0] )
        
        for i in raw_pred:
            
            if i >= 0:
                
                y_pred.append(1)
                
            elif i < 0:
                
                y_pred.append(-1)
                
        return y_pred
        

# Example Set
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],
])

y = np.array([-1,-1,1,1,1])    
df = pd.DataFrame(X,columns = ['One','Two','Three'])
df.insert(3,'Four',y)


        
        
        