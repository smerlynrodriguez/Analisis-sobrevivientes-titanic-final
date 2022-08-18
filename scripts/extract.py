import pandas as pd


def extractData(path):

	dataset = pd.read_csv(path)
	return dataset


def extractInfo(dataset):

	columns = dataset.columns
	shape = dataset.shape
	info = dataset.info()
	describe = dataset.describe()
	describe_categories = 	dataset.describe(include=['0'])
    
	return columns, shape, info, describe, describe_categories

def outputData(dataset, y_pred, name):
    
    output = pd.DataFrame({'PassengerId': dataset.PassengerId, 'Survived': y_pred})
    output.to_csv(name, index=False)
	