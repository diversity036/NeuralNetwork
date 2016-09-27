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
		self.rate = 0.01          		# learning rate

		b1 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfInput)
		b2 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfOutput)

		self.w1 = np.random.uniform(-b1, b1, [self.NumOfHidden, self.NumOfInput])
		# w1: weight input -> hidden layer, w1[i][j]: jth input -> ith hidden layer
		self.w2 = np.random.uniform(-b2, b2, [self.NumOfOutput, self.NumOfHidden])
		# w2: weight hidden layer -> output, w2[i][j]: jth hidden layer -> ith output		

	def Sigmoid(self, aij):
		## we use sigmoid as the activation function in every hidden layer
		## aj: the linear combination of the input
		## zj: the output after activation
		zij = 1/(1 + np.exp(-aij))
		return zij


	def Softmax(self, a):
		output = np.exp(a)
		return	output/sum(output)
		 

	def BackProp(self, train_file_name):
		error = 0
		for l in open(train_file_name).read().splitlines():

			## read the txt file: target, input(784)
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			#inputs = np.append([1], np.array(map(float, line)))
			inputs = np.array(map(float, line))

			## compute output for hidden layer and store it in a1
			a1 = self.Sigmoid(np.dot(self.w1, inputs))
			
			## compute output for output layer and store it in a2
			output = np.dot(self.w2, a1)
			
			## compute softmax
			a2 = self.Softmax(output)
			
			## now doing backprop, hidden -> output
			## first comput target 
			t = np.zeros(self.NumOfOutput)
			t[target] = 1
			t = self.Softmax(t)

			error += -self.Error(t, a2)
			# print "error", error
	
			deltak = (t-a2)*a2*(1-a2)
			
			## input -> hidden
			deltaj = np.dot(deltak, self.w2)*a1*(1-a1)
			
			## update w1 & w2
			# for x2 in range(self.w2.shape[0]):
			# 	for y2 in range(self.w2.shape[1]): 
			# 		self.w2[x2][y2] += deltak[x2]*a1[y2]*self.rate 
			
			self.w2 += np.tile(deltak, (self.NumOfHidden,1)).transpose()*np.tile(a1, (self.NumOfOutput, 1))*self.rate
			
			# for x1 in range(self.w1.shape[0]): 
			# 	for y1 in range(self.w1.shape[1]):
			# 		#print deltaj[x1]*inputs[y1]*self.rate
			# 		self.w1[x1][y1] += deltaj[x1]*inputs[y1]*self.rate

			self.w1 += np.tile(deltaj, (self.NumOfInput,1)).transpose()*np.tile(inputs, (self.NumOfHidden, 1))*self.rate

		return error/3000
	
	def Error(self, t, a):
		
		return	sum(t*np.log(a))
		



	def trainNN(self):
		self.Initialization()
		n = 0
		start_time = time.time()
		while(n < 200):
			error = self.BackProp(sys.argv[1])
			print "fin_error", error
			n += 1
		return error
		print("--- %s seconds ---" % (time.time() - start_time))
				


if __name__ == "__main__":

	NN = NeuralNetwork()
	error = NN.trainNN()
	print "Error after 200 iteration: ", error

	
