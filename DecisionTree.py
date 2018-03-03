# Run this program on your local python
# interpreter, provided you have installed
# the required libraries.
 
# Importing the required packages
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt 

def importdata():
    x = np.arange(0,50)
    x = pd.DataFrame({'x':x})

    # just random uniform distributions in differnt range
    y1 = np.random.uniform(10,15,10)
    y2 = np.random.uniform(20,25,10)
    y3 = np.random.uniform(0,5,10)
    y4 = np.random.uniform(30,32,10)
    y5 = np.random.uniform(13,17,10)
    y = np.concatenate((y1,y2,y3,y4,y5))
    return x, y
 
# Function to split the dataset
def splitdataset(X, Y):
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
     
    return X, Y, X_train, X_test, y_train, y_test
     
def train_using_regression(X_train, X_test, y_train):
    reg_tree = DecisionTreeRegressor(max_depth = 2);
    reg_tree.fit(X_train, y_train)
    return reg_tree
 
# Boosted Decision Trees
    # predictions by ith decisision tree
def train_using_BoostedDecisionTree(x, y):
    xi = x # initialization of input
    yi = y # initialization of target
    # x,y --> use where no need to change original y
    ei = 0 # initialization of error
    n = len(yi)  # number of rows
    predf = 0 # initial prediction 0
    trees = []
    for i in range(30): # loop will make 30 trees (n_estimators). 
        tree = DecisionTreeRegressor(max_depth = 3)
        tree.fit(xi, yi)
        trees.append(tree)
        predf = tree.predict(xi);
        ei = yi - learningRate*predf  # needed originl y here as residual always from original y    
        yi = ei # update yi as residual to reloop
    return trees;

# Function to make predictions
def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred

def predictionUsingBDT(X_test, trees):
    xlen = len(X_test)
    predictVal = [0]*xlen
    
    for tree in trees:
        predictVal = predictVal + learningRate*tree.predict(X_test)
    return predictVal;

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print ("Accuracy : ", mean_absolute_error(y_test, y_pred))

     
def main():
    # Building Phase
    X, Y = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(X, Y)
    tree = train_using_regression(X_train, X_test, y_train)
    
    global learningRate 
    learningRate = 1
    trees = train_using_BoostedDecisionTree(X_train, y_train)
     
    # Prediction using one tree
    y_1 = prediction(X_test, tree)
    cal_accuracy(y_test, y_1)
    # Prediction using 30 trees
    y_2 = predictionUsingBDT(X_test, trees)
    cal_accuracy(y_test, y_2)

    X = X_test
    plt.figure()
    plt.plot(X, y_test, 'ko', label="training samples")
    plt.plot(X, y_1, 'bo',  label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, 'r+', label="n_estimators=30", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()
     
# Calling main function
if __name__=="__main__":
    main()