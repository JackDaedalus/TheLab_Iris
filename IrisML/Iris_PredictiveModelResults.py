##

## B8IT10N Data Analytics  Machine Learning Workflow Program

## February 2020 + March 2020 in Visual Studio

## Ciaran Finnegan - Student No. 10524150

## Python Functions to read dataset and perform various descriptive and visual analysis routines

##

## Module Imports for Machine Learning in Python
from sklearn import metrics


## Module Imports for custom Python code to evaluate and compare modelling algorithms
from Iris_AlgorithmEvaluationAndComparison import *



def EvaluateAndPredictiveModel(X_train, Y_train, X_test, Y_test, dataDescription):
    
    # Fit the chosen model - SVM in this case - to the training dataset and make predictions
    # against the test data, which was 'he;d back'
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    Y_predictions = model.predict(X_test)

    # Evaluate and present the predictions generated in the model and compared to the expected
    # results in the test dataset
    print("\n\nEvaluate and present results for preditive model against {} test data : \n".format(dataDescription))
    print("\n\tPrediction Accuracy (metrics) : ", metrics.accuracy_score(Y_test, Y_predictions))

    print("\n\tConfusion Matrix : \n\t {0}".format(metrics.confusion_matrix(Y_test, Y_predictions)))

    print("\n\tClassification Report\n")
    print(metrics.classification_report(Y_test, Y_predictions))
    print("\n")

    
    conf_mat = metrics.confusion_matrix(Y_test, Y_predictions)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(conf_mat,annot=True, fmt='.2f', xticklabels=True, yticklabels=True, cmap="OrRd")
    # Format Correlation table for improved display
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_yticklabels(ax.get_yticklabels(), va="center", rotation = 0)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual class")
    plt.show()

    # Code for use in two class problems - 'positives' and 'negatives'
    #print('Confusion matrix: \n', conf_mat)
    #print('TP: ', conf_mat[1,1])
    #print('TN: ', conf_mat[0,0])
    #print('FP: ', conf_mat[0,1])
    #print('FN: ', conf_mat[1,0])






