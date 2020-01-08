
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



def PreSplitDataManipulation(dataset, datasetDescription, sDFColumnNames, sClassCol):

    # Run routines to identify potential data quality issues in the rows of the dataset
    CheckQualityOfDataset(dataset, datasetDescription, sDFColumnNames)

    # Converting Categorical features into Numerical features
    final_data = ConvertCategoricalFeaturesInDataset(dataset, datasetDescription, sClassCol)



    return final_data



def CheckQualityOfDataset(dataset, datasetDescription, sColNames):

    # Check for Null Values
    print("\n\tChecking for Null Values in {} Dataset - Result : {}\n".format(datasetDescription, dataset.isnull().values.any()))
    # Pause
    #anykey = input("\nPress any key..")

    # Check for Duplicates
    numOfDuplicatedRows = dataset.duplicated().value_counts()
    print("\n\tChecking for Duplicate Rows in {} Dataset - Result : {}\n\n".format(datasetDescription, numOfDuplicatedRows))
    # Pause
    #anykey = input("\nPress any key..")

    # Check for hidden missing (zero) values
    sDatasetColNames = sColNames
    arrFeatureCheckListForZeroValues = sDatasetColNames
    print("\n\t# Rows in {1} dataframe {0}".format(len(dataset), datasetDescription))
    for feature in arrFeatureCheckListForZeroValues:
        print("\n\t# zero value rows in column {1}: {0}".format(len(dataset.loc[dataset[feature] == 0]),feature))


def ConvertCategoricalFeaturesInDataset(dataset, datasetDescription, sClass):

    # Converting Categorical features into Numerical features - most algorithms need numeric values
    # Just one column - the 'Flower Class' needs to be converted from a Categorical Values
    # This is the target variable and an 'Iris-setosa' is assigned a value of '2', and 'Iris-versicolor' is assigned
    # a value of '1', and Iris-virginica is assigned the default of '3'
    print("\n\n\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(dataset.head(2))

    # Pause
    #anykey = input("Press any key..")
    
    dataset[sClass] = dataset[sClass].apply(ConvertCategoricalFeatures)
    dfConvertedDS = dataset

    # Display the first two rows after conversion of 'Tree Type'
    print("\nCategorical {} Dataset Head Rows Prior to Flower Calss Type conversion : \n".format(datasetDescription))
    print(dfConvertedDS.head(2))

    # Pause
    #anykey = input("Press any key..")
    

    # Check for Correlation after all features converted to numeric
    CheckDatasetForCorrelation(dfConvertedDS, datasetDescription + " (AFTER Categorical Conversion)")


    return dfConvertedDS



