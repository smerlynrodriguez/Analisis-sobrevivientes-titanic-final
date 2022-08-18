import matplotlib.pyplot as plt 
import extract
import transform
import model



def main():

	train = extract.extractData("../data/train.csv")
	test = extract.extractData("../data/train.csv")

	print(f'Encabezado del dataset: {test}')
	print(f'el dataset contiene las siguentes columlas: {extract.extractData.columns(train)}')
	print(f'La dimencion de los datos es de {extract.extraData.shape}')

	print(f'Los sobrevivientes del barco de distribuian de la siguente forma: ')
	print(transform.EdaAnalyze(train).survivors_per_sex)


if __name__ == "__main__":
	main()




