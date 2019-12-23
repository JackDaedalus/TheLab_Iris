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



def MainProg_CATwo():

	# Set PyCharm Display Option 
	# This is done to improve console display
	# for use in documentented screen shots
	desired_width=320
	pd.set_option('display.width', 400)
	np.set_printoptions(linewidth=10)
	pd.set_option('display.max_columns',15)


	# Identify file to be read into dataset
	filename = "Spruce.csv"
	# Set up file identifier for use in Console Print statements
	dataDescription = "Spruce Dataset"

	# Read CSV file and return dataset
	df_spruce = ReadDataframe(filename)

	# Display some basic initial statistics about dataset
	# This data will be used to inform follow up data cleansing actions
	DisplayBasicDataFrameInfo(df_spruce, dataDescription)

	df_FinalSpruce = PreSplitDataManipulation(df_spruce, dataDescription)

	X, Y = CreateLableAndFeatureSet(df_FinalSpruce, dataDescription)

	X_train, X_test, Y_train, Y_test, X_Scaled = CreateTrainingAndTestData(X, Y, dataDescription, df_FinalSpruce)

	#TuneRandomForestAlgorithm(X_train, Y_train)

	# Determined by algorithm tuning process
	best_estimator = 550

	ImplementTunedRandomForestAlgorithm(X_train, X_test, Y_train, Y_test, best_estimator, X)

	X = CreateRevisedFeatureSet(X)

	## --- Rinse / Repeat ##
	X_train, X_test, Y_train, Y_test, X_Scaled = CreateTrainingAndTestData(X, Y, dataDescription, df_FinalSpruce)
	#####

	# ---- Call RandomForest with revised FeatureSet
	ImplementTunedRandomForestAlgorithm(X_train, X_test, Y_train, Y_test, best_estimator, X)


	### ---- Implement PCA Visualisation and K-Means Clustering
	x_pca = ImplementPCAVisualisation(X_Scaled, Y, dataDescription)

	ImplementK_MeansClustering(X_Scaled, x_pca, dataDescription)



def CreateTrainingAndTestData(X, Y, dataDescription, origDataset):

	X_Scaled = NormaliseTrainingData(X, dataDescription)

	X_train, X_test, Y_train, Y_test = SplitDatasetIntoTrainAndTestSets(X_Scaled, Y, dataDescription, origDataset)

	X_train, Y_train = ImplementOverSampling(X_train, Y_train)


	return X_train, X_test, Y_train, Y_test, X_Scaled



def ReadDataframe(filename):

	print("\n\tReading {} file..\n".format(filename))

	# Read CSV file into panda dataframe
	df = pd.read_csv(filename)

	# Return the panda dataframe read in from the CSV file
	return df 


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


def ConvertTreeType(column):

	# Converting Categorical features into Numerical features
    if column == 'Spruce':
        return 1
    else:
        return 0

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
	# Just one column - the 'Tree Type' needs to be converted from a Categorical Values
	# This is the target variable and a 'Spruce' is assigned a value of '1', and 'Other' is assigned 
	# a value of '0' 
	# Display the first two rows after conversion of 'Tree Type'
	print("\nCategorical {} Dataset Head Rows Prior to Tree Type conversion : \n".format(datasetDescription))
	print(dataset.head(2))

	dataset['Tree_Type'] = dataset['Tree_Type'].apply(ConvertTreeType)
	final_data = dataset

	# Display the first two rows after conversion of 'Tree Type'
	print("\nConverted Categorical {} Dataset Head Rows : \n".format(datasetDescription))
	print(final_data.head(2))
	# Pause
	anykey = input("Press any key..")

	# Display the change in datatype for 'Tree Type'
	print("\nConverted Categorical {} Dataset Datatypes : \n".format(datasetDescription))
	print(final_data.dtypes)
	# Pause
	anykey = input("Press any key..")


	# Pre-Split Data Preparation
	# Hidden missing values - check the zeroes - we already checked for NULL
	#print(final_data.head(10))
	#Elevation	Slope	Horizontal_Distance_To_Hydrology	Vertical_Distance_To_Hydrology	Horizontal_Distance_To_Roadways	Horizontal_Distance_To_Fire_Points
	SpruceFeatureCheckListForZeroValues = ['Elevation','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']
	print("\n\t# Rows in {1} dataframe {0}".format(len(final_data), datasetDescription))
	# It would not seem logical that any of the first six colums in the Spruce dataset have a zero value
	# This loop checks if there are any zero values
	# If there were any zero values the user would determine the appropriate follow up action
	for feature in SpruceFeatureCheckListForZeroValues:
		print("\n\t# zero value rows in column {1}: {0}".format(len(final_data.loc[final_data[feature] == 0]),feature))
	
	# Pause
	anykey = input("Press any key..")

	# Drop rows?

	# Check for Correlation after all features converted to numeric
	CheckDatasetForCorrelation(final_data, datasetDescription)


	return final_data




def CheckDatasetForCorrelation(dataset, dataDescription):

	print("\n\tCheck {} Dataset For any Correlation between features (Categorical features converted into Numerics): \n".format(dataDescription))

	# Correlation analysis - a graphical representation of possible correlation of data
	sns.heatmap(dataset.corr(), annot=True, fmt='.2f')
	# Pause
	anykey = input("Press any key..")



	
def CreateLableAndFeatureSet(final_data, dataDescription):

	# Check distribution of survived and died in cleaned dataset
	# Check that the observation of any outcome is not too rare
	num_obs = len(final_data)
	num_true = len(final_data.loc[final_data['Tree_Type'] == 1]) # Spruce Tree = True
	num_false = len(final_data.loc[final_data['Tree_Type'] == 0]) # Spruce Tree = False
	print("Number of Spruce Tree Types :  {0} ({1:2.2f}%)\n".format(num_true, (num_true/num_obs) * 100))
	print("Number of Other Tree Types : {0} ({1:2.2f}%)\n".format(num_false, (num_false/num_obs) * 100))


	# Dividing dataset into label and feature sets
	X = final_data.drop('Tree_Type', axis = 1) # Features
	Y = final_data['Tree_Type'] # Labels
	
	print("\n\tDimentions of Label and Feature Dataset for {}".format(dataDescription))
	print(X.shape)
	print(Y.shape)
	print("\n\tFeatured + Labeled - {} Dataset Head Rows X + Y : \n".format(dataDescription))
	print(X.head(2))
	print(Y.head(2))
	# Pause
	#anykey = input("Press any key..")

	return X, Y


def NormaliseTrainingData(X, dataDescription):

	print("\n\tScaling the Feature dataset..")

	# Normalizing numerical features so that each feature has mean 0 and variance 1
	feature_scaler = StandardScaler()
	X_scaled = feature_scaler.fit_transform(X)
	#print("\nPre-Scaled Features - {} Dataset Head Rows : \n".format(dataDescription))
	#print(X.head(3))
	# Pause
	#anykey = input("Press any key..")
	#print("\nPost-Scaled Features - {} Dataset Head Rows : \n".format(dataDescription))
	#print(X_scaled.view())
	# Pause
	#anykey = input("Press any key..")

	return X_scaled


def SplitDatasetIntoTrainAndTestSets(X_Scaled, Y, dataDescription, final_data):

	# Dividing dataset into training and test sets
	X_train, X_test, Y_train, Y_test = train_test_split(X_Scaled, Y, test_size = 0.3, random_state = 100)
	print("\n\t{} Training Set Shape : \n".format(dataDescription))
	print(X_train.shape)
	print("\n\t{} Test Set Shape : \n".format(dataDescription))
	print(X_test.shape)


	# Need to check if we have the desired 70 / 30 split in Train and Test Data
	print("\n\t{0:0.2f}% in {1} training set".format( (len(X_train)/len(final_data.index)) * 100, dataDescription))
	print("\n\t{0:0.2f}% in {1} test set".format((len(X_test)/len(final_data.index)) * 100, dataDescription))


	# Verifying predicted value was split correctly - according to proportion in original dataset
	print("\n")
	print("\tOriginal True  : {0} ({1:0.2f}%)".format(len(final_data.loc[final_data['Tree_Type'] == 1]), (len(final_data.loc[final_data['Tree_Type'] == 1])/len(final_data.index)) * 100.0))
	print("\tOriginal False : {0} ({1:0.2f}%)".format(len(final_data.loc[final_data['Tree_Type'] == 0]), (len(final_data.loc[final_data['Tree_Type'] == 0])/len(final_data.index)) * 100.0))
	print("\n")
	print("\tTraining True  : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 1]), (len(Y_train[Y_train[:] == 1])/len(Y_train) * 100.0)))
	print("\tTraining False : {0} ({1:0.2f}%)".format(len(Y_train[Y_train[:] == 0]), (len(Y_train[Y_train[:] == 0])/len(Y_train) * 100.0)))
	print("\n")
	print("\tTest True      : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 1]), (len(Y_test[Y_test[:] == 1])/len(Y_test) * 100.0)))
	print("\tTest False     : {0} ({1:0.2f}%)".format(len(Y_test[Y_test[:] == 0]), (len(Y_test[Y_test[:] == 0])/len(Y_test) * 100.0)))
	print("\n")



	return X_train, X_test, Y_train, Y_test


def ImplementOverSampling(X_train,Y_train):

	# Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
	print("Number of observations in each class before oversampling (training data): \n", pd.Series(Y_train).value_counts())

	smote = SMOTE(random_state = 101)
	X_train,Y_train = smote.fit_sample(X_train,Y_train)

	print("Number of observations in each class after oversampling (training data): \n", pd.Series(Y_train).value_counts())
	# Pause
	# anykey = input("Press any key..")

	return X_train,Y_train



def TuneRandomForestAlgorithm(X_train, Y_train):

	"""
	In the below GridSearchCV(), scoring parameter should be set as follows:
	scoring = 'accuracy' when you want to maximize prediction accuracy
	scoring = 'recall' when you want to minimize false negatives
	scoring = 'precision' when you want to minimize false positives
	scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
	"""

	# Tuning the random forest parameter 'n_estimators' using Grid Search
	rfc = RandomForestClassifier(criterion='entropy', max_features='auto', random_state=1)
	scoreOptions = ['accuracy','recall','precision','f1']
	#grid_param = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]}
	grid_param = {'n_estimators': [10,20]}

	print("\n\tRunning Grid Search Cross Validation tuning..")

	for score in scoreOptions:
		gd_sr = GridSearchCV(estimator=rfc, param_grid=grid_param, scoring=score, cv=5)

		print("\n\tScore paramter set to : {}".format(score))
		# Execute the fit function on the Training Data
		gd_sr.fit(X_train, Y_train)

		# Determine and display the optimum hyperparameter
		best_parameters = gd_sr.best_params_
		print("\n\tOptimum parameter is : {}".format(best_parameters))
		#print(best_parameters)

		best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
		print("\n\tOptimum scoring result is : {}".format(best_result))
		#print(best_result)

	# Pause
	anykey = input("Press any key..")


def ImplementTunedRandomForestAlgorithm(X_train, X_test, Y_train, Y_test, best_estimator, X):

	# Building random forest using the tuned parameter
	#rfc = RandomForestClassifier(n_estimators=400, criterion='entropy', max_features='auto', random_state=1)
	rfc = RandomForestClassifier(n_estimators=best_estimator, criterion='entropy', max_features='auto', random_state=1)
	rfc.fit(X_train,Y_train)
	
	# Rate the importance of the features to guide the creation of a more targeted feature set for the algorithm
	featimp = pd.Series(rfc.feature_importances_, index=list(X)).sort_values(ascending=False)
	print("\n\tList of features in dataset by importance to prediction model : \n")
	print(featimp)
	# Pause
	# anykey = input("Press any key..")

	Y_pred = rfc.predict(X_test)
	print("\n\tPrediction Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))


	# Displaying a Confusion Matrix
	# Text on screen
	print("\n\tConfusion Matrix\n")
	print("{0}".format(metrics.confusion_matrix(Y_test, Y_pred)))
	print("\n")
	print("\n\tClassification Report\n")
	print(metrics.classification_report(Y_test, Y_pred))
	# Pause
	# anykey = input("Press any key..")


	conf_mat = metrics.confusion_matrix(Y_test, Y_pred)
	plt.figure(figsize=(8,6))
	sns.heatmap(conf_mat,annot=True)
	plt.title("Confusion_matrix")
	plt.xlabel("Predicted Class")
	plt.ylabel("Actual class")
	plt.show()
	print('Confusion matrix: \n', conf_mat)
	print('TP: ', conf_mat[1,1])
	print('TN: ', conf_mat[0,0])
	print('FP: ', conf_mat[0,1])
	print('FN: ', conf_mat[1,0])


def CreateRevisedFeatureSet(dataset):

	# Selecting features with higher sifnificance and redefining feature set
	X = dataset[['Elevation', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Hydrology','Slope','Soil_Type20','Soil_Type21','Soil_Type9','Soil_Type27','Soil_Type36']]

	# Check for Correlation after reduced feature set created
	CheckDatasetForCorrelation(X, "Reduced Spruce Trees Feature Set")


	return X



def ImplementPCAVisualisation(X_Scaled, Y, dataDescription):

	# Implementing PCA to visualize dataset
	print("\n\tThe Implementation of PCA Visualisation...\n")

	pca = PCA(n_components = 2)
	pca.fit(X_Scaled)
	x_pca = pca.transform(X_Scaled)
	print(pca.explained_variance_ratio_)
	print(sum(pca.explained_variance_ratio_))

	# Pause
	# anykey = input("Press any key..")

	plt.figure(figsize = (8,6))
	plt.scatter(x_pca[:,0], x_pca[:,1], c=Y, cmap='plasma')
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

	return x_pca


def ImplementK_MeansClustering(X_Scaled, x_pca, dataDescription):

	# Implementing K-Means CLustering on dataset and visualizing clusters
	print("\n\tThe Implementation of K-Means Clustering and Visualisation...\n")

	# Finding the number of clusters (K)
	inertia = []
	for i in range(1,11):
		kmeans = KMeans(n_clusters = i, random_state = 100)
		kmeans.fit(X_Scaled)
		inertia.append(kmeans.inertia_)

	plt.plot(range(1, 11), inertia)
	plt.title('The Elbow Plot')
	plt.xlabel('Number of clusters')
	plt.ylabel('Inertia')
	plt.show()

	kmeans = KMeans(n_clusters = 2)
	kmeans.fit(X_Scaled)
	print(kmeans.cluster_centers_)
	plt.figure(figsize = (8,6))
	plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans.labels_, cmap='plasma')
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()


MainProg_CATwo()