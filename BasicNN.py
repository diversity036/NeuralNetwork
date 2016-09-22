import sys
import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork():

	def Initialization(self):

		self.NumOfHidden = 100          # num of hidden layer
		self.NumOfOutput = 10			# num of output layer
		self.NumOfInput  = 784			# num of input
		self.rate = 0.1					# learning rate




	def FormatData(self, filename):
		for l in open(filename).read().splitlines():
			line = l.split(',')
			label = line[-1]
	
	def Cal_Sigmoid(self, aj):
		## we use sigmoid as the activation function in every hidden layer
		## aj: the linear combination of the input
		## zj: the output after activation
		zj = 1/(1 + np.exp(-aj))
		return zj

	def Cal_Softmax(self, )
			


	


if __name__ == "__main__":

	NN = NeuralNetwork()

	
