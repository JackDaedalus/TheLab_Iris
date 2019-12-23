
# A test Machine Learning program using the Iris dataset
# 17th December 2019



# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy as np
print('numpy: {}'.format(np.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas as pd
print('pandas: {}'.format(pd.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))



# Load libraries
import seaborn as sns
import matplotlib.pyplot as plt
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




def MainProg_IrisML():

    
    # Set up file identifier for use in Console Print statements
    sDataDescription = "Iris Flower Dataset" 
    

    # Load Iris Dataset
    df_Iris = LoadIrisDataset()

    # Display some basic initial statistics about dataset
	# This data will be used to inform follow up data cleansing actions
    DisplayBasicDataFrameInfo(df_Iris,sDataDescription)

    # Extend analysis of data set with visualisations
    DisplayDataVisualization(df_Iris,sDataDescription)
       

    # Dummy Function used to avoid format errors in main function
    DummyEndFunc(sDataDescription)




def DummyEndFunc(sDatasetDescription):

    print("\n\n\n")
    print("\n\tTerminating the Python analysis for : {}..\n".format(sDatasetDescription))


    


def LoadIrisDataset():
    
    # Load dataset

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    print("\n\tLoading the Dataset from URL : {}..\n".format(url))
    
    dataset = read_csv(url, names=names)

    return dataset



def DisplayBasicDataFrameInfo(df, sDatasetDescription):

    print("\n\tDisplay the Dataset descriptions for : {}..\n".format(sDatasetDescription))

    # shape
    print("\n\tDisplay the Dataset shape for : {}..\n".format(sDatasetDescription))
    print(df.shape)
    # head
    print("\n\tDisplay the Dataset head for : {}..\n".format(sDatasetDescription))
    print(df.head(20))
    # descriptions
    print("\n\tDisplay the Dataset description for : {}..\n".format(sDatasetDescription))
    print(df.describe())
    # class distribution
    print("\n\tDisplay the Dataset class distribution for : {}..\n".format(sDatasetDescription))
    print(df.groupby('class').size())



def DisplayDataVisualization(df, sDatasetDescription):

    CheckDatasetForCorrelation(df, sDatasetDescription)
    PrintCorrelationGraphic(df, sDatasetDescription)



def CheckDatasetForCorrelation(dataset, dataDescription):

	print("\nCheck {} Dataset For any Correlation between features : \n".format(dataDescription))
	print(dataset.corr())

    # Correlation analysis - a graphical representation of possible correlation of data
	# sns.heatmap(dataset.corr(), annot=True, fmt='.2f')
    # Pause
	anykey = input("Press any key..")


def PrintCorrelationGraphic(dataset, dataDescription):

    print("\nDisplay Correlation matrix for Dataset {} : \n".format(dataDescription))
    plt.matshow(dataset.corr())
    plt.show()

    # Pause
    anykey = input("Press any key..")



    f, ax = plt.subplots(figsize=(10, 8))
    corr = dataset.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
    
    # Pause
    anykey = input("Press any key..")


     



    




# Launch the Iris ML main function - call the logic from here
MainProg_IrisML()
