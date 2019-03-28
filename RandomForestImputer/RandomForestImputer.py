import pandas as pd
import numpy as np
from MachineLearningAlgorithms.RandomForest import RandomForest

class RandomForestImputer:
    
    def __init__(self,n_tree = 10):
        
        self.n_tree = n_tree

    def _initial_replace(self,df_m,missing_column):
        
        m,n = df_m.shape
        
        # Find to column of the missing data
        for i in range(0,n):
            
            if pd.isna(df_m.iloc[-1,i]):
                missing_column = i
                break
            
        if isinstance(df_m.iloc[0,missing_column],str):
            
            replaced_df = df_m.fillna(df_m.iloc[:,missing_column].mode()[0])
        
        else:
            
            replaced_df = df_m.fillna(df_m.iloc[:,missing_column].median())
    
        return replaced_df
    
    def _proximity_matrix(self,df):
        
        m = df.shape[0]
        
        rf = RandomForest(n_tree = self.n_tree)
        
        rf.train(df)
        
        
        p_matrix = np.zeros((m-1,self.n_tree))
                
        for i in range(0,rf.n_tree):
            
            prediction = rf.trees[i].predict(df)
            
            for j in range(0,m-1):
                
                    if (prediction[j] == prediction[-1]):
                        
                        p_matrix[j,i] += 1
        
        p_matrix = (np.sum(p_matrix, axis = 1))/self.n_tree
        
        p_matrix = pd.DataFrame(p_matrix)
        
        p_matrix.index = df.index[:-1]
        
        return p_matrix
    
    def _replace(self,df_one_missing,proximity_matrix,missing_column,missing_index):
        
        if isinstance(df_one_missing.iloc[0,missing_column],str):
            
            values, occurance = np.unique(df_one_missing.iloc[:-1,missing_column],
                                          return_counts = True)
            
            total_occurance = np.sum(occurance)
            total_proximity = np.sum(proximity_matrix.values)
            
            selection_list = list()
            
            for i in range(0,len(values)):
                
                indices = (df_one_missing.iloc[:-1,missing_column] == values[i])
                prob = np.sum(proximity_matrix[indices].values) / total_proximity
                
                selection_list.append( (occurance[i]/total_occurance) * (prob) )
            
            max_value = np.array(selection_list).max()
            
            max_value_index = np.where(np.array(selection_list)==max_value)
            
            df_one_missing.iloc[-1,missing_column] = values[max_value_index]
            
            return df_one_missing.iloc[-1,missing_column]
        
        else:
            
            total_proximity = np.sum(proximity_matrix.values)
            s = (proximity_matrix.values/total_proximity).T @ df_one_missing.iloc[:-1,missing_column].values
            return s[0]

    def impute(self,df):
                
        m,n = df.shape
        self.df = df
        
        total_miss = df.isnull().sum().sum()
        #Check if there is a missing value in columns and record the ones have
        missing_value_columns = list() 
        
        for i in range(0,n):
            
            if np.any(df.iloc[:,i].isna()):
                
                missing_value_columns.append(i)
        

        for i in missing_value_columns:
            
            # Create a copy of the given DataFrame and drop all missing values to be used
            # during the training of the trees
            df_no_missing = df.dropna()
                        
            missing_value_indices = np.array([])
            
            missing_value_index = df[pd.isna(df.iloc[:,i])].index.values
            
            missing_value_indices = np.append(missing_value_indices, missing_value_index)
            
            for j in missing_value_indices.astype(int):
                
                df_one_missing = df_no_missing.append(df.iloc[j,:])
                
                df_initial_replaced = self._initial_replace(df_one_missing,i)
                
                proximity_matrix = self._proximity_matrix(df_initial_replaced)
                
                self.replaced_element = self._replace(df_one_missing,proximity_matrix,i,j)
                
                self.df.iloc[j,i] = self.replaced_element
                
                print(f"Remaining Missing Value: {self.df.isnull().sum().sum()} / {total_miss}")
             
        return self.df
    
    def _create_df_for_each_class(self,df_one_missing):
        
        unique_classes = np.unique(df_one_missing.iloc[:-1,-1])
        
        n_classes = len(unique_classes)
        
        return_dfs = list()
        
        for i in unique_classes:
            
            df_one_missing.iloc[-1,-1] = i
            
            return_dfs.append(df_one_missing.copy())
        
        return return_dfs
    
    def impute_test_data(self,df_train,df_test):
        
        mt,nt = df_train.shape
        
        me,ne = df_test.shape
        
        missing_value_columns = list()  
        
        for i in range(0,ne):
            
            if np.any(df_test.iloc[:,i].isna()):
                
                missing_value_columns.append(i)
        
        for i in missing_value_columns:
            
            dft_no_missing = df_train.dropna()
            
            missing_value_indices = np.array([])
            
            missing_value_index = df_test[pd.isna(df_test.iloc[:,i])].index.values
            
            missing_value_indices = np.append(missing_value_indices, missing_value_index)  
            
            for j in missing_value_indices.astype(int):
                
                changed_value = list()
                                        
                dft_one_missing = dft_no_missing.append(df_test.iloc[j,:])
                
                df_list = self._create_df_for_each_class(dft_one_missing)
                
                counts = np.zeros((len(df_list)))
                
                for k in range(0,len(df_list)):
                    
                    r_df = self.impute(df_list[k].reset_index(drop = True))
                    
                    changed_value.append(self.replaced_element)
                    
                    rf = RandomForest(n_tree = self.n_tree)
                    
                    rf.train(r_df)
                    
                    for l in range(0,self.n_tree):
                        
                        pred = rf.trees[l]._predict_for_one(r_df.iloc[-1,:])
                        
                        if pred == r_df.iloc[-1,-1]:
                            counts[k] += 1
                
                max_index = np.argmax(counts)
               
                replace_value = changed_value[max_index]
            
                df_test.iloc[j,i] = replace_value
                
                print(f"Remaining Missing Value: {df_test.isnull().sum().sum()}")
               
        self.rdf = df_test
                    
                    
                    
                    