import pandas as pd
import numpy as np
from MachineLearningAlgorithms.RandomForest.RandomForest import RandomForest

class RandomForestImputer:
    
    def __init__(self, n_tree = 10, n_repetition = 2):
        
        self.n_tree = n_tree # The number of trees during imputation
        self.n_repetition = n_repetition # The number of repetition of whole process
                                         # higher the number, missing values converge better
                                         # better educated guess
    # This function primitively changes missing value pointed in missing_column
    # With using missing_column, last row of the missing column is a missing sample because
    # of implementations explained ahead.
    # If the missing value is string change the value with the mode of the corresponding column
    # If the missing value is numeric change the value with median of the corresponding column
        
        self.dummy = 0 # Dummy variable to prevent overlapping print statements
    def _initial_replace(self,df_m,missing_column):
        
        m,n = df_m.shape

        if isinstance(df_m.iloc[0,missing_column],str):
            
            replaced_df = df_m.fillna(df_m.iloc[:,missing_column].mode()[0])
        
        else:
            
            replaced_df = df_m.fillna(df_m.iloc[:,missing_column].median())
    
        return replaced_df
    
    
    # Calculates the proximities of the missing value to the other samples of the set

    # Algorithm is as follows:
        # 1- Grow a random forest with given n_tree number of trees
        # 2- Create a matrix where rows are the samples in the data set except the missing one
        # 3- Columns are the decision trees in the forest
        # 4- For each tree, add 1 to the rows when missing data (initially replaced one) 
        #    and the corresponding sample are ended up predicting same value (proximity values)
        # 5- Sum the proximity values along the tree axis to determine proximity of each sample
        #    with the missing value and normalize the result
    
    def _proximity_matrix(self,df):
        
        # Create a Random Forest and train it
        m = df.shape[0]
        
        rf = RandomForest(n_tree = self.n_tree)
        
        rf.train(df)
        
        # Initialize proximity matrix
        p_matrix = np.zeros((m-1,self.n_tree))
        
        # For each tree in the forest and for each sample calculate proximities with the
        # initially replaced missing value
        for i in range(0,rf.n_tree):
            
            prediction = rf.trees[i].predict(df)
            
            for j in range(0,m-1):
                
                    if (prediction[j] == prediction[-1]):
                        
                        p_matrix[j,i] += 1
        
        # Normalize the tree and convert it into dataframe for easy exploration
        p_matrix = (np.sum(p_matrix, axis = 1))/self.n_tree
        
        p_matrix = pd.DataFrame(p_matrix)
        
        p_matrix.index = df.index[:-1]
        
        return p_matrix
    
    # Main replacing algorithm:
    # 1- If the missing value belongs to categorical column, count each categorical variable's
    #    occurance
    # 2- For each unique categorical variable calculate probability of occurance using
    #    proximity matrix (sum of values in proximity matrix of the unique variable / total sum
    #    of the proximity matrix)
    # 3- Multiply the result of step 2 with occurance probability of the unique variable
    #    Overall formula can be given as:
    #    Sum of proximity matrix which results in given unique category / Sum of all proximity matrix
    #    * (Number of occurance of the given unique variable / total number of samples)
    #    Store the results in selection_list list as scoreboard
    # 4- Repeat step 2 for each variable, replace the missing value with highest score
    # 5- If the missing value belongs to numeric column, take the weighted sum of of each sample
    #    Weight determined by (proximity matrix value of the sample / sum of all proximity matrix)
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
    
    # Main function to handle whole process
    def impute(self,df):
                
        m,n = df.shape
        self.df = df
        
        # Record total of missing values for printing purposes
        total_miss = df.isnull().sum().sum()
        
        #Check if there is a missing value in columns and record the ones have
        missing_value_columns = list() 
        
        for i in range(0,n):
            
            if np.any(df.iloc[:,i].isna()):
                
                missing_value_columns.append(i)
        
        # Repeat for each missing column
        for i in missing_value_columns:
            
        # Create a copy of the given DataFrame and drop all missing values to be used
        # during the training of the trees
            df_no_missing = df.dropna()
                        
            missing_value_indices = np.array([]) # Initialize missing value indices array
            
            # Find indices of missing values for the ith missing value column
            missing_value_index = df[pd.isna(df.iloc[:,i])].index.values
            
            # Add all the missing indices of the ith missing column to the initialized empty
            # array
            missing_value_indices = np.append(missing_value_indices, missing_value_index)
            
            # Repeat for each missing indices in the ith column
            for j in missing_value_indices.astype(int):
                
                # Append the row which have missing value to the array with no missing values
                df_one_missing = df_no_missing.append(df.iloc[j,:])
                
                # Primitively impute the missing value  
                df_initial_replaced = self._initial_replace(df_one_missing,i)
                
                # Repeat the whole process for user-specified number of times to better 
                # educated guess
                # Note: imputed dataset becomes initially_replaced after first iteration
                for k in range(0,self.n_repetition):
                    
                    # Calculate proximity matrix for imputed dataset
                    proximity_matrix = self._proximity_matrix(df_initial_replaced)
                
                    # Replace the missing element according to the proximity matrix
                    self.replaced_element = self._replace(df_one_missing,proximity_matrix,i,j)
                
                    self.df.iloc[j,i] = self.replaced_element
                    
                    # Change the initialy replaced dataframe with imputed one for repetition
                    df_initial_replaced.iloc[-1,:] = self.df.iloc[j,:]
            
            # Use dummy varible to prevent printing of this section when imputing test set
            if self.dummy == 0:
                print(f"Remaining Missing Value: {self.df.isnull().sum().sum()} / {total_miss}")
            else:
                pass
        return self.df
    
    ## Only for imputation of the test set, for given dataframe which contains single missing
    # sample, create list of datasets for each unique outputs
    
    # Example
    # We have a testing dataset with no output labels, some biological information as features
    # and unique labels are whether the person has cancer or not
    # For the row contains missing value, create 2 datasets which for the first one, missing row
    # labeled as has cancer, second one labeled as doesn't have cancer
    # Note: Unique labels are determined from the training set
    def _create_df_for_each_class(self,df_one_missing):
        
        # Determine unique classes in the given set
        unique_classes = np.unique(df_one_missing.iloc[:-1,-1])
    
        # Calculate number of unique classes
        n_classes = len(unique_classes)
        
        # Initialize empty list for the data sets
        return_dfs = list()
        
        # For each unique output label create a copy of the data set and label the missing 
        # value for each unique label (number of datasets = number of unique output label)
        for i in unique_classes:
            
            df_one_missing.iloc[-1,-1] = i
            
            return_dfs.append(df_one_missing.copy())
        
        return return_dfs
    
    ## Main imputing algorithm for the test sets
    # Algorithm is as follows;
    
    # 1- Acquire training set, find a missing row and append to the end of the training set
    # 2- Create list of datasets for each unique label in the training set and impute the
    #    output part of the missing data with those
    # 3- Apply same algorithm for imputing training set to all datasets, replace the missing
    #    value 
    # 4- For each dataframe run a random forest algorithm to classify for the missing value row
    # 5- Impute the missing value with highest scored dataset
    def impute_test_data(self,df_train,df_test):
        
        self.dummy = 1
        
        mt,nt = df_train.shape
        
        me,ne = df_test.shape
        
        missing_value_columns = list()  
        
        # Find missing columns and indices using same algorithm given in imputing training set
        for i in range(0,ne):
            
            if np.any(df_test.iloc[:,i].isna()):
                
                missing_value_columns.append(i)
        
        for i in missing_value_columns:
            
            dft_no_missing = df_train.dropna()
            
            missing_value_indices = np.array([])
            
            missing_value_index = df_test[pd.isna(df_test.iloc[:,i])].index.values
            
            missing_value_indices = np.append(missing_value_indices, missing_value_index)  
            
            for j in missing_value_indices.astype(int):
                
                # Create a candidate variable list for imputation
                changed_value = list()
                
                dft_one_missing = dft_no_missing.append(df_test.iloc[j,:])
                
                # Create list of dataframes for each unique variable
                df_list = self._create_df_for_each_class(dft_one_missing)
                
                # 
                counts = np.zeros((len(df_list)))
                
                # Repeat for each dataframe in the dataframe list
                for k in range(0,len(df_list)):
                    
                    # Impute the missing value with using original imputing algorithm
                    r_df = self.impute(df_list[k].reset_index(drop = True))
                    
                    # Add the imputing variable as candidate to the candidate list
                    changed_value.append(self.replaced_element)
                    
                    # Grow a random forest with using imputed dataframe 
                    rf = RandomForest(n_tree = self.n_tree)
                    
                    rf.train(r_df)
                    
                    # Predict for the missing row, if correctly predicted increase count matrix
                    # (performance matrix) by 1 for this dataframe 
                    for l in range(0,self.n_tree):
                        
                        pred = rf.trees[l]._predict_for_one(r_df.iloc[-1,:])
                        
                        if pred == r_df.iloc[-1,-1]:
                            counts[k] += 1
                
                # From the list of candidate imputing variable, choose the one with highest score
                # (score is determined by prediction performance on the missing row (counts matrix))
                max_index = np.argmax(counts)
               
                replace_value = changed_value[max_index]
            
                df_test.iloc[j,i] = replace_value
                
                print(f"Remaining Missing Value: {df_test.isnull().sum().sum()}")
               
        self.rdf = df_test
                    
                    
                    
                    