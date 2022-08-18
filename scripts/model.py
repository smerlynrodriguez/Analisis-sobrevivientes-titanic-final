import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import plot_confusion_matrix 


def ProcessingModel(table):
    
    Y_table = table['Survived']
    features = ['Sex', 'Age', 'Pclass', 'FlagSolo']
    X_table = table[features]
    logistig_regretion = LogisticRegression()
    logistig_regretion = logistig_regretion.fit(X_table, Y_table)
    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(X_table, Y_table)
    
    return Y_table, features, X_table, logistig_regretion, decision_tree


def ModelEvaluation(model, Y_table, X_table):
    
    disposition = plot_confusion_matrix(model, X_table, Y_table, cmap=plt.cm.Blues, values_format="d")
    true_prediction = disposition.confusion_matrix[0,0]+disposition.confusion_matrix[1,1]
    total_data = np.sum(disposition.confusion_matrix)
    accuracy = true_prediction/total_data

    return disposition, true_prediction, accuracy, total_data


def ModelTest(dataset, features, Y_table, X_table, logistig_regretion, decision_tree):
    
	dataset_info = (dataset.info())
	dataset_features = dataset.features[features]
	dataset_test_set = dataset_features.shape
	Y_logistig_regretion = logistig_regretion.predict(Y_table)
	Y_pred_tree = decision_tree.predict(X_table)

	return dataset_info, dataset_features, dataset_test_set, Y_logistig_regretion, Y_pred_tree