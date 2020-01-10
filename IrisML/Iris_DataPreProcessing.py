
##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##


## Module Imports for custom Python code to read and analyse dataset
from Iris_LoadandAnalyseData import *

## Module Imports for Machine Learning in Python
from sklearn.preprocessing import StandardScaler


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
    dsConvertedDataset = ConvertCategoricalFeaturesInDataset(dataset, datasetDescription, sClassCol)

    # Check distribution of Classification / Label Data
    CheckDatasetDistribution(dsConvertedDataset, datasetDescription, sClassCol, 3)


    final_data = dsConvertedDataset


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
    # CheckDatasetForCorrelation(dfConvertedDS, datasetDescription + " (AFTER Categorical Conversion)")


    return dfConvertedDS




def CheckDatasetDistribution(dataset, datasetDescription, sClass, iNumOfClasses):

    # Check distribution of classifications / labels
    # Check that the observation of any outcome is not too rare
    num_obs = len(dataset)

    # Print distribution percentages
    print("\n\n\nDistribution of Classification/Label data in {} Dataset : \n".format(datasetDescription))
    iCount = 0 
    while iCount < iNumOfClasses:

        num_class = len(dataset.loc[dataset[sClass] == iCount]) # Iterate through classifications
        print("Number of Classification {2} : {0} ({1:2.2f}%)\n".format(num_class, (num_class/num_obs) * 100, iCount))
        iCount += 1




def CreateLableAndFeatureSet(final_data, dataDescription, sClass):

    # Dividing dataset into label and feature sets
    X = final_data.drop(sClass, axis = 1) # Features
    Y = final_data[sClass] # Labels

    print("\n\tDimentions of Label and Feature Dataset for {}".format(dataDescription))
    print(X.shape)
    print(Y.shape)
    print("\n\tFeatured + Labeled - {} Dataset Head Rows X + Y : \n".format(dataDescription))
    print(X.head(2))
    print(Y.head(2))

    # Normalise the data to prevent large differences in absolute feature values skewing the model
    # results
    X_scaled = X
    #X_scaled = NormaliseTrainingData(X, dataDescription)


    return X_scaled, Y



def NormaliseTrainingData(X, dataDescription):

    print("\n\tScaling the Features for {} dataset..".format(dataDescription))

    # Normalizing numerical features so that each feature has mean 0 and variance 1
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    #print("\nPre-Scaled Features - {} Dataset Head Rows : \n".format(dataDescription))
    #print(X.head(3))

    #print("\nPost-Scaled Features - {} Dataset Head Rows : \n".format(dataDescription))
    #print(X_scaled.view())


    return X_scaled