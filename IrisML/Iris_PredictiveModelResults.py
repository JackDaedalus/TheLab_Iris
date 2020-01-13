##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## January 2019 

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##

## Module Imports for Machine Learning in Python


## Module Imports for custom Python code to evaluate and compare modelling algorithms
from Iris_AlgorithmEvaluationAndComparison import *



def EvaluateAndPredictiveModel(X_train, Y_train, X_test, Y_test, dataDescription):
    
    # Fit the chosen model - SVM in this case - to the training dataset and make predictions
    # against the test data, which was 'he;d back'
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)

    # Evaluate and present the predictions generated in the model and compared to the expected
    # results in the test dataset
    print("\n\nEvaluate and present results for preditive model against {} test data : \n".format(dataDescription))
    print(accuracy_score(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))

