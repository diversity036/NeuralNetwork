import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import time

class NeuralNetwork():

	def Initialization(self):

		self.NumOfHidden = 100          # num of hidden layer
		self.NumOfOutput = 10			# num of output layer
		self.NumOfInput  = 784			# num of input
		self.rate = 0.01          		# learning rate
		self.w1 = {} 					# w1: weight input -> hidden layer
		self.w2 = {} 					# w1: weight hidden layer -> output
		
		b1 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfInput)
		b2 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfOutput)

		for x1 in range(self.NumOfHidden): 
			self.w1[x1] = {}
			for y1 in range(self.NumOfInput):
				self.w1[x1][y1] = np.random.uniform(-b1, b1)

		for x2 in range(self.NumOfOutput):
			self.w2[x2] = {}
			for y2 in range(self.NumOfHidden): 
				self.w2[x2][y2] = np.random.uniform(-b2, b2)


	def Sigmoid(self, aij):
		## we use sigmoid as the activation function in every hidden layer
		zij = 1/(1 + np.exp(-aij))
		return zij

	def Softmax(self, a):
		output = {}
		
		for key_a in a:
			output[key_a] = np.exp(a[key_a])

		sum_a = sum(output.values())
		for key_o in output:
			output[key_o] /= sum_a

		return output


	def BackProp(self, train_file_name):
		error = 0
		for l in open(train_file_name).read().splitlines():

			## read the txt file: target, input(784)
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			
			inputs = {}
			for i in range(self.NumOfInput):
				inputs[i] = float(line[i])

			## compute output for hidden layer and store it in a1
			a1 = {}
			for i, val in self.w1.iteritems():
				sum_1 = 0
				for j in val:
					sum_1 += inputs[j]*val[j]
				a1[i] = self.Sigmoid(sum_1)
			

			## compute output for output layer and store it in a2
			output = {}
			for i, val in self.w2.iteritems():
				sum_2 = 0
				for j in val:
					sum_2 += a1[j]*val[j]
				output[i] = sum_2
			
			## compute softmax
			a2 = self.Softmax(output)

			## now doing backprop, hidden -> output
			## first comput target
			t = {}
			for i in range(self.NumOfOutput):
				if (i== target): 
					t[i] = 1
				else:
					t[i] = 0
			t = self.Softmax(t)

			error += self.Error(t, a2)

			deltak = {}
			for k in range(self.NumOfOutput):	
				deltak[k] = (t[k]-a2[k])*a2[k]*(1-a2[k])

			## input -> hidden
			deltaj = {}
			for j in range(self.NumOfHidden):
				sum_3 = 0
				for k in range(self.NumOfOutput):
					sum_3 += deltak[k]*self.w2[k][j]
				deltaj[j] = sum_3*a1[j]*(1-a1[j])
			
			## update w1 & w2
			for x2 in range(self.NumOfOutput):
				for y2 in range(self.NumOfHidden): 
					self.w2[x2][y2] += deltak[x2]*a1[y2]*self.rate 

			for x1 in range(self.NumOfHidden): 
				for y1 in range(self.NumOfInput):
					self.w1[x1][y1] += deltaj[x1]*inputs[y1]*self.rate

		return error/3000


	def Error(self, t, a):
		error = 0
		for i in t:
			error += -t[i]*np.log(a[i])
		return error

	def Train(self):
		self.Initialization()
		n = 0
		while(n < 200):
			start_time = time.time()
			error = self.BackProp(sys.argv[1])
			print "final", error
			print("--- %s seconds ---" % (time.time() - start_time))
			n += 1


if __name__ == "__main__":
	
	NN = NeuralNetwork()
	NN.Train()
	