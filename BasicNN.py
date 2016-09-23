## This is the code for a single hidden layer with backprop ##
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
		self.w1 = np.zeros([self.NumOfHidden, self.NumOfInput+1])
		# w1: weight input -> hidden layer, w1[i][j]: jth input -> ith hidden layer
		self.w2 = np.zeros([self.NumOfOutput, self.NumOfHidden+1])
		# w1: weight hidden layer -> output, w1[i][j]: jth hidden layer -> ith output
		
		b1 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfInput)
		b2 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfOutput)

		for x1 in range(self.w1.shape[0]): 
			for y1 in range(self.w1.shape[1]):
				self.w1[x1][y1] = np.random.uniform(-b1, b1)

		for x2 in range(self.w2.shape[0]):
			for y2 in range(self.w2.shape[1]): 
				self.w2[x2][y2] = np.random.uniform(-b2, b2)

	
	def Sigmoid(self, aij):
		## we use sigmoid as the activation function in every hidden layer
		## aj: the linear combination of the input
		## zj: the output after activation
		zij = 1/(1 + np.exp(-aij))
		return zij


	def Softmax(self, a):
		output = np.zeros(a.size)
		for i, ai in enumerate(a):
			output[i] = np.exp(ai)
		return	output/sum(output)
		 

	def BackProp(self, train_file_name):
		for l in open(train_file_name).read().splitlines():

			## read the txt file: target, input(784+1)
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			inputs = np.append([1], np.array(map(float, line)))

			## compute output for hidden layer and store it in a1
			a1 = np.zeros(self.NumOfHidden)
			for i, wi in enumerate(self.w1):
				a1[i] = self.Sigmoid(np.dot(inputs, wi))

			a1 = np.append([1], a1);

			## compute output for output layer and store it in a2
			a2 = np.zeros(self.NumOfOutput)
			for j, wj in enumerate(self.w2):
				a2[j] = np.dot(a1, wj)

			## compute softmax
			output = self.Softmax(a2)

			
				


if __name__ == "__main__":

	NN = NeuralNetwork()
	NN.Initialization()
	NN.BackProp(sys.argv[1])

	
