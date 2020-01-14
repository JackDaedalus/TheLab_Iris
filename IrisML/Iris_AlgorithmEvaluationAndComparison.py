
##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##

## Module Imports for Machine Learning in Python
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

## Module Imports for custom Python code to read and analyse dataset
from Iris_LoadandAnalyseData import *




def CreateTrainingAndTestData(X_Scaled, Y, dataDescription):

	X_train, X_test, Y_train, Y_test = SplitDatasetIntoTrainAndTestSets(X_Scaled, Y, dataDescription)

	X_train, Y_train = ImplementOverSampling(X_train, Y_train)


	return X_train, X_test, Y_train, Y_test



def SplitDatasetIntoTrainAndTestSets(X_Scaled, Y, dataDescription):

	# Dividing dataset into training and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X_Scaled, Y, test_size = 0.20, random_state = 1)
	print("\n\t{} Training Set Shape : \n".format(dataDescription))
	print(X_train.shape)
	print("\n\t{} Test Set Shape : \n".format(dataDescription))
	print(X_test.shape)


	return X_train, X_test, Y_train, Y_test



def ImplementOverSampling(X_train, Y_train):
    return X_train, Y_train


# Evaluate different algorithm models
def EvaluateAndCompareAlgorithms(X_train, Y_train, dataDescription):
    
    # Set up algorithms
    models = []
    models.append(('Log Regression ', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    models.append(('K-Nearest Neighbors ', KNeighborsClassifier()))
    models.append(('Classification and Regression Trees ', DecisionTreeClassifier()))
    models.append(('Gaussian Naive Bayes ', GaussianNB()))
    models.append(('Support Vector Machines ', SVC(gamma='auto')))
    
    # Test Values and evaluation metric
    print("\n\n\tBuild and evaluate models for {} dataset : \n".format(dataDescription))
    results = []
    names = []
    iCount = 1
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print('\n\t%s %s: %f (%f)' % (iCount, name, cv_results.mean(), cv_results.std()))
        iCount +=1

    # Visual Comparison of Algorithms
    # plt.boxplot(results, labels=names)
    # plt.title('Algorithm Comparison')
    # plt.show()