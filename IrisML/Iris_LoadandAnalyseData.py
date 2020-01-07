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



def ReadDataframe():

    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    print("\n\tLoading the Dataset from URL : {}..\n".format(url))
    
    dataset = pd.read_csv(url, names=names)


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



def CheckDatasetForCorrelation(dataset, dataDescription):

    print("\n\tCheck {} Dataset For any Correlation between features (Categorical features converted into Numerics): \n".format(dataDescription))
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