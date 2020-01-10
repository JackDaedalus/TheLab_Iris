##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##


## Module Imports for Machine Learning in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix




def ReadDataframe():

    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    arrColNames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    sDSClassCol = 'class'
    print("\n\tLoading the Dataset from URL : {}..\n".format(url))
    
    dataset = pd.read_csv(url, names=arrColNames)

   
    return dataset, arrColNames, sDSClassCol



def DisplayBasicDataFrameInfo(dataset, datasetDescription, sClass):
    
    # Shape
    print("\n\t{} Dataset Dimensions (Shape): \n".format(datasetDescription))
    print(dataset.shape)

    # Display Head rows in dataset
    print("\n\t{} Dataset Head Rows : \n".format(datasetDescription))
    print(dataset.head())

    # Describe - present a summary of dataset attributes
    print("\n\t{} Dataset 'Describe()' : \n".format(datasetDescription))
    print(dataset.describe())

    # dTtpes - present the data types for each attribute in the dataset
    print("\n\t{} Dataset Datatypes : \n".format(datasetDescription))
    print(dataset.dtypes)

    # Class distribution - present the count of the number of rows that 
    # belong to each class in the dataset
    print("\n\t{} Dataset Class Distribution : \n".format(datasetDescription))
    print(dataset.groupby(sClass).size())

    # Info
    print("\n\t{} Dataset 'Info()' : \n".format(datasetDescription))
    print(dataset.info())

    # Check for Correlation before all features converted to numeric
    CheckDatasetForCorrelation(dataset, datasetDescription + " (BEFORE Categorical Conversion)")




def CheckDatasetForCorrelation(dataset, dataDescription):

    print("\n\tCheck {} Dataset For any Correlation between features : \n".format(dataDescription))
    print(dataset.corr())
    # Correlation analysis - a graphical representation of possible correlation of data
    plt.figure(figsize=(12,8))
    ax = sns.heatmap(dataset.corr(), annot=True, fmt='.2f', xticklabels=True, yticklabels=True)
    # Format Correlation table for improved display
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_yticklabels(ax.get_yticklabels(), va="center", rotation = 0)
    # Set Correlation Table Descriptions
    sCorrTableTitle = ("Correlation Table for {} Dataset\n".format(dataDescription))
    plt.title(sCorrTableTitle)
    plt.xlabel("\nX Label\n")
    plt.ylabel("\nY Label\n")
    plt.show()




def DisplayVisualDataFrameInfo(dataset, dataDescription):
    
    # Display Histograms for each attribute in dataset
    dataset.hist()
    plt.show()

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()




