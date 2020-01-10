##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Added to GitHub - January 2019

## Main Python Program

##



## Module Imports for Machine Learning in Python
 
# Python version
import sys

import scipy
import numpy
import matplotlib
import pandas
import sklearn

# Check the versions of libraries
# Print values to console for verification
#print('Python: {}'.format(sys.version))
#print('scipy: {}'.format(scipy.__version__))
#print('numpy: {}'.format(numpy.__version__))
#print('matplotlib: {}'.format(matplotlib.__version__))
#print('pandas: {}'.format(pandas.__version__))
#print('sklearn: {}'.format(sklearn.__version__))


# Load libraries

from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


## Module Imports for Python GUI widgets
from Iris_GUI import *

## Module Imports for custom Python code to read and analyse dataset
from Iris_LoadandAnalyseData import *

## Module Imports for custom Python code to pre-process dataset before modelling
from Iris_DataPreProcessing import *

## Module Imports for custom Python code to evaluate and compare modelling algorithms
from Iris_AlgorithmEvaluationAndComparison import *



def Main_IrisML():
    # Set PyCharm Display Option
    # This is done to improve console display
    # for use in documentented screen shots
    desired_width = 320
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=10)
    pd.set_option('display.max_columns', 15)

    # Set up file identifier for use in Console Print statements and graphical output
    sDatasetDescription = "Flower Iris"
    #sDatasetDescription = GetDatasetDescription()

    # Read CSV file and return dataset
    df_Iris, dfColNames, sDSClassCol = ReadDataframe()

    # Display basic initial statistics about dataset
    # This data will be used to inform follow up data cleansing actions
    # DisplayBasicDataFrameInfo(df_Iris, sDatasetDescription, sDSClassCol)

    # Display visual representations of the dataset attributes
    # These representations will also help with decisions on pre-modellling
    # data manipulation and algorithm selection / execution
    # DisplayVisualDataFrameInfo(df_Iris, sDatasetDescription)

    # Amend the Dataset so that modelling algorithms can be successfully applied
    df_FinalIris = PreSplitDataManipulation(df_Iris, sDatasetDescription, dfColNames, sDSClassCol)

    # Divide dataset into label and feature sets (feature set is standardised)
    X_Scaled, Y = CreateLableAndFeatureSet(df_FinalIris, sDatasetDescription, sDSClassCol)

    # Split dataset into training and test data
    X_train, X_test, Y_train, Y_test= CreateTrainingAndTestData(X_Scaled, Y, sDatasetDescription)

    # Evaluate different algorithm models
    EvaluateAndCompareAlgorithms(X_train, Y_train, sDatasetDescription)




Main_IrisML()