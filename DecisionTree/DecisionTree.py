import numpy as np
import pandas as pd

class DecisionTree:
    
    def __init__(self, min_samples = 2, max_depth=999):
        
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.counter = 0
        self.best_entropy = 10
        
    def _ispure(self,data):
        
        y = data[:,-1]
        # If output column contains only one class return true otherwise return false
        if len( np.unique(y) ) == 1:
            
            return True
        
        else:
            
            return False
    

    def _classify(self,data):
    
        y = data[:,-1]
        # Procedure to find mode of the output array, return mode value
        classes, occurance = np.unique(y, return_counts = True)
        
        # Sometimes there is a problem when tree returns a empty array to classify which
        # raises an eror, in order to work around that if such error occurs return a random
        # value which doesn't affect the overall tree
        try:
            return classes[np.argmax(occurance)]
        
        except:
            
            return 1

        
    
    def _all_possible_splits(self,data):
        
        # Takes average of each 2 unique sample of a feature and calls it a potential split
        # returns a dic which shows each possible split for each feature
        
        # If the feature is categorical returns each categorical variable as a potential 
        # split
        splits = {}
        
        for i in range(0,data.shape[1] - 1):
            
 
            splits[i] = []
            
            data_unique = np.unique(data[:,i])
            
            if isinstance(data[0,i],str):
                
                for j in range(0,data_unique.shape[0]):
                    
                    splits[i].append(data_unique[j])
            else:
                
                for j in range(1,data_unique.shape[0]):

                    splits[i].append( (data_unique[j] + data_unique[j-1]) / 2)
                
        return splits     
    
    def _split_data(self,data,column,value):
        
        # Splits data in 2 parts, 1- greater part of the data from the given column 
        # and value, 2- less part of the given split
        # if the given column is categorical, splits into; 1- part of data equal to given
        # column and value, 2- not equal part
        if isinstance(data[0,column],str):
            
            data_greater_indices = data[:,column] == value
            data_less_indices = data[:,column] != value
        
        else:
            data_greater_indices = data[:,column] > value
            data_less_indices = data[:,column] <= value
    
        return data[data_greater_indices], data[data_less_indices]
    

    def _calculate_entropy(self,data_higher,data_lower):
        
        # Calculates occurrence of each class in the data_higher part
        occurrence_higher = np.unique( data_higher[:,-1], return_counts = True)[1]
        
        # Calculates possibility of occurrunce for each class in the data_higher part
        prob_higher = occurrence_higher / sum(occurrence_higher)
        
        # Calculates entropy of the data_higher part
        entropy_higher = sum( prob_higher * -np.log2(prob_higher) )
        
        # Repeat same procedure for data_lower part
        occurrence_lower = np.unique( data_lower[:,-1], return_counts = True)[1]
        
        prob_lower = occurrence_lower / sum(occurrence_lower)
        
        entropy_lower = sum( prob_lower * -np.log2(prob_lower) )
        
        
        # Calculate overall entropy
        
        total_higher = data_higher.shape[0]
        total_lower = data_lower.shape[0]
        
        # Calculate probability for lower and higher sets 
        prob_total_higher = (total_higher) / (total_higher + total_lower)
        prob_total_lower = (total_lower) / (total_higher + total_lower)
        
        overall_entropy = prob_total_higher * entropy_higher + prob_total_lower * entropy_lower
        
        return overall_entropy
        
    def _find_best_split(self,data):
        
        # Initial best_entropy, just a random value
        self.best_entropy = 99999999
        
        # Find all possible splits for given data
        splits = self._all_possible_splits(data)
        
        
        # Repeat for each key and each value of the dict acquired from _all_possible_splits
        # function
        
        for i in splits:
            
            for j in splits[i]:
                
                # Splits data into greater and lower for given ith column's jth value
                greater_data, lower_data = self._split_data(data,i,j)
                
                # Calculate overall entropy for given splits 
                current_entropy = self._calculate_entropy(greater_data, lower_data)
                
                
                # Algorithm to choose best split depending on the entropy
                if current_entropy <= self.best_entropy:
                    
                    self.best_entropy = current_entropy
                    self.best_column_split = i
                    self.best_value_split = j
        
    
        return self.best_column_split, self.best_value_split
    
    
    def train(self,data):
        
        
        # Convert dataframe to numpy array, because of recursion it'll raise and error
        # as a workaround there is a try - except module
        try:
            
            self.column_names = data.columns
            data = data.values
        
        except:
            
            pass
        
        
        # Stopping criteria for recursion
        
        # 1- If the given data part contains only 1 class
        # 2- If given maximum depth is exceeded
        # 3- If number of rows in the given data is less then min_samples
        if ( self._ispure(data) ) or (self.max_depth <= self.counter) or (self.min_samples > data.shape[0]):
            
            classification = self._classify(data)
            return classification
        
        
        # If the criteria weren't satisfied
        else:
            
            # Increase a counter for maximum depth purposes
            self.counter = self.counter + 1
            
            # Find best column and best_value to split
            best_column, best_value = self._find_best_split(data)
            
            # Split data into 2 parts from given best split
            data_greater, data_less = self._split_data(data,best_column,best_value)
            
            # If at least one of the splitted data is empty, go the base case 
            if len(data_greater) == 0 or len(data_less) == 0:
                
                classification = self._classify(data)
                
                return classification
            
            # If the split occured in a categorical column, question will be is the data
            # equal (yes_answer) or not equal (no_answer) to the best_value
            if isinstance(best_value, str):
                
                question =  f"{self.column_names[best_column]} = {best_value}"
            
            # If the split occured in a numerical column, question will be is the data
            # greater (yes_answer) or less (no_answer) than the best_value
            else:
                
                question =  f"{self.column_names[best_column]} >= {best_value}"
            
            # Add question to the dictionary as a key 
            tree = {question : []}
            
            # Call function for the greater or equal to parts
            yes_answer = self.train(data_greater)
            
            # Call function for the less or not equal to parts
            no_answer = self.train(data_less)
            
            if yes_answer == no_answer:
                tree = yes_answer
            else:
                tree[question].append(yes_answer)
                tree[question].append(no_answer)
            
            # Append each yes and no answers as a value for the question key

        
        # Create a key variable for the class for explore purposes
        self.tree = tree
        
        # Copy same tree for further use in prediction part
        self.test_tree = tree

        return tree

    # This function predicts the result for one row 
    def _predict_for_one(self,test_sample):
        
        # Acquire the question from the tree
        question = list(self.test_tree.keys())[0]
        
        # Acquire the column name which split will be held
        column_name = list(self.test_tree.keys())[0].split(' ')[0]
        
        # Obtain the value of given question from the tree
        if isinstance( test_sample.loc[column_name], str):
            
            value = list(self.test_tree.keys())[0].split(' ')[2]
        
        else:
            
            value = float(list(self.test_tree.keys())[0].split(' ')[2])
        
        
        # 1- If the value is a string, check if the given column data is equal or not 
        # to the value
        # 2- If the value is a numeric, check if the data in the given column
        # is greater or less than the value
        # Select the answer (0th index for yes, 1st index for no)
        if isinstance(value,str):
            
            if test_sample[column_name] == value:
        
                answer = self.test_tree[question][0]
                
            else:
                
                answer = self.test_tree[question][1]
        else:
            
            if test_sample[column_name] >= value:
            
                answer = self.test_tree[question][0]
        
            else:
            
                answer = self.test_tree[question][1]
        
        # If the answer acquired is a dictionary (another question), call function from
        # that question, if the answer is a string return the string as a answer 
        # Repeat recursively until reaching a string value which is the prediction
        if isinstance( answer, dict):
            
            self.test_tree = answer
            return self._predict_for_one(test_sample)
            
        else:
            
            self.test_tree = self.tree
            return answer
    
    # Prediction for all rows of given data, repeats _predict_for_one for number of rows
    # and store the results in an array
    def predict(self,test_data):
    
        predictions = []
        
        for sample_n in range(0, test_data.shape[0]):
            
            prediction = self._predict_for_one(test_data.iloc[sample_n])
            predictions.append(prediction)
        
        return np.array(predictions)
