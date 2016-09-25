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
		self.rate = 0.1          		# learning rate
		self.w1 = np.zeros([self.NumOfHidden, self.NumOfInput])
		self.dw1 = np.zeros([self.NumOfHidden, self.NumOfInput])
		# w1: weight input -> hidden layer, w1[i][j]: jth input -> ith hidden layer
		self.w2 = np.zeros([self.NumOfOutput, self.NumOfHidden])
		self.dw2 = np.zeros([self.NumOfOutput, self.NumOfHidden])
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
			#inputs = np.append([1], np.array(map(float, line)))
			inputs = np.array(map(float, line))

			## compute output for hidden layer and store it in a1
			a1 = np.zeros(self.NumOfHidden)
			for i, wi in enumerate(self.w1):
				a1[i] = self.Sigmoid(np.dot(inputs, wi))
			#print "a1", a1
			#a1 = np.append([1], a1);

			## compute output for output layer and store it in a2
			output = np.zeros(self.NumOfOutput)
			for j, wj in enumerate(self.w2):
				output[j] = np.dot(a1, wj)
			#print output, "output"
			## compute softmax
			a2 = self.Softmax(output)

			#print "output: ", a2

			## now doing backprop, hidden -> output
			## first comput target
			#print target 
			t = np.zeros(self.NumOfOutput)
			for i in range(self.NumOfOutput):
				if (i== target): 
					t[i] = 1
			t = self.Softmax(t)

			deltak = np.zeros(self.NumOfOutput)
			for k in range(self.NumOfOutput):	
					deltak[k] = (t[k]-a2[k])*a2[k]*(1-a2[k])
		
			#print "deltak", deltak

			## input -> hidden
			deltaj = np.zeros(self.NumOfHidden)
			for j in range(self.NumOfHidden):
				deltaj[j] = np.dot(deltak, self.w2[:,j])*a1[j]*(1-a1[j])
			#print deltaj
			## update w1 & w2
			for x2 in range(self.w2.shape[0]):
				for y2 in range(self.w2.shape[1]): 
					# print "delta", deltak[x2]
					# print "a1", a1[y2]
					# print "------------"
					self.w2[x2][y2] += deltak[x2]*a1[y2]*self.rate 

			for x1 in range(self.w1.shape[0]): 
				for y1 in range(self.w1.shape[1]):
					#print deltaj[x1]*inputs[y1]*self.rate
					self.w1[x1][y1] += deltaj[x1]*inputs[y1]*self.rate
			
			print self.w2[0][0]
				


if __name__ == "__main__":

	NN = NeuralNetwork()
	NN.Initialization()
	NN.BackProp(sys.argv[1])

	
