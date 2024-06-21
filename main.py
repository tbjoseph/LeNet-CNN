from solution import *
from helper import *


if __name__ == '__main__':

	data_dir = '../cifar-10-batches-py/'

	print('Loading and preprocessing...')
	x_train, y_train, x_test, y_test = load_data(data_dir)
	x_train, x_test = preprocess(x_train, x_test, normalize=True)
	x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

	model = LeNet_Cifar10(n_classes=10)
	model.train(x_train, y_train, x_valid, y_valid, 128, 20)

	accuracy = model.test(x_test, y_test)
	with open('test_result.txt', 'w') as f:
		f.write(str(accuracy))
	
	print('Test accuracy: %.4f' %accuracy)