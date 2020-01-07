
##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##




## Module Imports for custom Python code to read and analyse dataset
from Iris_LoadandAnalyseData import *



def ConvertCategoricalFeatures(column):

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
    #anykey = input("\nPress any key..")

    # Check for Duplicates
    numOfDuplicatedRows = dataset.duplicated().value_counts()
    print("\n\tChecking for Duplicate Rows in {} Dataset - Result : {}\n\n".format(datasetDescription, numOfDuplicatedRows))
    # Pause
    #anykey = input("\nPress any key..")

    # Converting Categorical features into Numerical features - most algorithms need numeric values
    # Just one column - the 'Flower Class' needs to be converted from a Categorical Values
    # This is the target variable and an 'Iris-setosa' is assigned a value of '2', and 'Iris-versicolor' is assigned
    # a value of '1', and Iris-virginica is assigned the default of '3'
    print("\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(dataset.head(2))


    # Pause
    #anykey = input("Press any key..")
    
    dataset['class'] = dataset['class'].apply(ConvertCategoricalFeatures)
    final_data = dataset

    # Display the first two rows after conversion of 'Tree Type'
    print("\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(final_data.head(2))

    # Pause
    #anykey = input("Press any key..")

    # Pre-Split Data Preparation
    # Hidden missing values - check the zeroes - we already checked for NULL



    # Drop rows?

    # Check for Correlation after all features converted to numeric
    CheckDatasetForCorrelation(final_data, datasetDescription + " (AFTER Categorical Conversion)")


    return final_data

