##

## B8IT106 Tools for Data Analytics  CA_TWO

## October 2019

## Added to GitHub - October 8th 2019

## October9 Branch Created

##


## Module Imports for Machine Learning in Python

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


##



def Main_IrisML():

    # Set PyCharm Display Option
    # This is done to improve console display
    # for use in documentented screen shots
    desired_width=320
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=10)
    pd.set_option('display.max_columns',15)


    # Set up file identifier for use in Console Print statements
    sDataDescription = "Irish Flower Dataset"

    # Read CSV file and return dataset
    df_Iris = ReadDataframe()

    # Display some basic initial statistics about dataset
    # This data will be used to inform follow up data cleansing actions
    DisplayBasicDataFrameInfo(df_Iris, sDataDescription)

    df_FinalSpruce = PreSplitDataManipulation(df_Iris, sDataDescription)






def ReadDataframe():

    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    print("\n\tLoading the Dataset from URL : {}..\n".format(url))

    filename = "Spruce.csv"
    
    dataset = pd.read_csv(url, names=names)
    #dataset = pd.read_csv(filename)

    return dataset


def DisplayBasicDataFrameInfo(dataset, datasetDescription):

    print("\n\t{} Dataset Head Rows : \n".format(datasetDescription))
    print(dataset.head())
    print("\n\t{} Dataset Dimensions : \n".format(datasetDescription))
    print(dataset.shape)
    print("\n\t{} Dataset Datatypes : \n".format(datasetDescription))
    print(dataset.dtypes)
    print("\n\t{} Dataset 'Info()' : \n".format(datasetDescription))
    print(dataset.info())
    print("\n\t{} Dataset 'Describe()' : \n".format(datasetDescription))
    print(dataset.describe())


def ConvertFlowerClass(column):

    # Converting Categorical features into Numerical features
    if column == 'Iris-setosa':
        return 0
    elif column == 'Iris-versicolor':
        return 1
    else:
        return 2



def PreSplitDataManipulation(dataset, datasetDescription):

    # Check for Null Values
    print("\n\tChecking for Null Values in {} Dataset - Result : {}\n".format(datasetDescription, dataset.isnull().values.any()))
    # Pause
    anykey = input("\nPress any key..")

    # Check for Duplicates
    numOfDuplicatedRows = dataset.duplicated().value_counts()
    print("\n\tChecking for Duplicate Rows in {} Dataset - Result : {}\n\n".format(datasetDescription, numOfDuplicatedRows))
    # Pause
    anykey = input("\nPress any key..")

    # Converting Categorical features into Numerical features - most algorithms need numeric values
    # Just one column - the 'Flower Class' needs to be converted from a Categorical Values
    # This is the target variable and an 'Iris-setosa' is assigned a value of '2', and 'Iris-versicolor' is assigned
    # a value of '1', and Iris-virginica is assigned the default of '3'
    print("\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(dataset.head(2))


    # Pause
    anykey = input("Press any key..")
    
    dataset['class'] = dataset['class'].apply(ConvertFlowerClass)
    final_data = dataset

    # Display the first two rows after conversion of 'Tree Type'
    print("\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(final_data.head(2))

    # Pause
    anykey = input("Press any key..")

    # Pre-Split Data Preparation
    # Hidden missing values - check the zeroes - we already checked for NULL



    # Drop rows?

    # Check for Correlation after all features converted to numeric
    CheckDatasetForCorrelation(final_data, datasetDescription)


    return final_data


def CheckDatasetForCorrelation(dataset, dataDescription):

    print("\n\tCheck {} Dataset For any Correlation between features (Categorical features converted into Numerics): \n".format(dataDescription))
    print(dataset.corr())
    # Correlation analysis - a graphical representation of possible correlation of data
    sns.heatmap(dataset.corr(), annot=True, fmt='.2f')
    # Pause
    anykey = input("Press any key..")


Main_IrisML()