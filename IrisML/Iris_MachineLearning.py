##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Added to GitHub - January 2019

## Main Python Program

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


## Module Imports for Python GUI widgets
from Iris_GUI import *

## Module Imports for custom Python code to read and analyse dataset
from Iris_LoadandAnalyseData import *

## Module Imports for custom Python code to pre=process dataset before modelling
from Iris_DataPreProcessing import *



def Main_IrisML():

    # Set up file identifier for use in Console Print statements and graphical output
    # sDatasetDescription = "Irish Flower"
    sDatasetDescription = GetDatasetDescription()

    # Read CSV file and return dataset
    df_Iris = ReadDataframe()

    # Display some basic initial statistics about dataset
    # This data will be used to inform follow up data cleansing actions
    DisplayBasicDataFrameInfo(df_Iris, sDatasetDescription)

    # Check for Correlation before all features converted to numeric
    CheckDatasetForCorrelation(df_Iris, sDatasetDescription + " (BEFORE Categorical Conversion)")

    df_FinalIris = PreSplitDataManipulation(df_Iris, sDatasetDescription)




Main_IrisML()