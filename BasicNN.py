## This is the code for a single hidden layer with backprop ##
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import argparse
class NeuralNetwork():

	def Initialization(self, args):

		self.NumOfHidden = args.hidden          # num of hidden layer
		self.NumOfOutput = 10			# num of output layer
		self.NumOfInput  = 784			# num of input
		self.NumOfTrain  = 3000			# num of training data
		self.NumOfValid  = 1000			# num of validation data
		self.NumOfTest   = 3000			# num of testing data
		self.rate        = args.rate          # learning rate
		self.NumOfEpoch  = args.epoch			# num of epoch
		self.train_err   = []
		self.valid_err   = []
		self.train_class_err = []
		self.valid_class_err = []
		self.dropout = args.dropout
		self.alpha = args.momentum
		c1 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfInput)
		c2 = np.sqrt(6)/np.sqrt(self.NumOfHidden + self.NumOfOutput)
		
		self.b1 = np.random.uniform(-c1, c1, self.NumOfHidden)
		self.b2 = np.random.uniform(-c2, c2, self.NumOfOutput)
		
		
		self.w1 = np.random.uniform(-c1, c1, [self.NumOfHidden, self.NumOfInput])
		# w1: weight input -> hidden layer, w1[i][j]: jth input -> ith hidden layer
		self.w2 = np.random.uniform(-c2, c2, [self.NumOfOutput, self.NumOfHidden])
		# w2: weight hidden layer -> output, w2[i][j]: jth hidden layer -> ith output
		self.dw1 = np.zeros(self.w1.shape)
		self.dw2 = np.zeros(self.w2.shape)		

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
		true_label = 0
		
		
		for l in open(train_file_name).read().splitlines():

			## read the txt file: target, input(784)
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			#inputs = np.append([1], np.array(map(float, line)))
			inputs = np.array(map(float, line))

			## compute output for hidden layer and store it in a1
			z1 = np.dot(self.w1, inputs)+ self.b1
			a1 = self.Sigmoid(z1)
			
			## compute output for output layer and store it in a2
			z2 = np.dot(self.w2, a1) + self.b2
			
			## compute softmax
			a2 = self.Softmax(z2)
			
			if (a2.argmax()==target):
				true_label += 1
			
			## now doing backprop, hidden -> output
			## first comput target 
			t = np.zeros(self.NumOfOutput)
			t[target] = 1
			#t = self.Softmax(t)

			error += -self.Error(t, a2)
			# print "error", error
	
			#deltak = (t-a2)*a2*(1-a2)
			
			delta2 = t - a2
			
			## input -> hidden
			delta1 = np.dot(delta2, self.w2)*a1*(1-a1)
			
			## update w1 & w2
			# for x2 in range(self.w2.shape[0]):
			# 	for y2 in range(self.w2.shape[1]): 
			# 		self.w2[x2][y2] += deltak[x2]*a1[y2]*self.rate 
			
			self.w2 += np.tile(delta2, (self.NumOfHidden,1)).transpose()*np.tile(a1, (self.NumOfOutput, 1))*self.rate + self.alpha*self.dw2
			self.dw2 = np.tile(delta2, (self.NumOfHidden,1)).transpose()*np.tile(a1, (self.NumOfOutput, 1))*self.rate d+ self.alpha*self.dw2
			
			
			# for x1 in range(self.w1.shape[0]): 
			# 	for y1 in range(self.w1.shape[1]):
			# 		#print deltaj[x1]*inputs[y1]*self.rate
			# 		self.w1[x1][y1] += deltaj[x1]*inputs[y1]*self.rate

			self.w1 += np.tile(delta1, (self.NumOfInput,1)).transpose()*np.tile(inputs, (self.NumOfHidden, 1))*self.rate + self.alpha*self.dw1
			self.dw1 = np.tile(delta1, (self.NumOfInput,1)).transpose()*np.tile(inputs, (self.NumOfHidden, 1))*self.rate + self.alpha*self.dw1
			
			self.b2 += delta2
			self.b1 += delta1
			
		return error/self.NumOfTrain, 1-true_label/float(self.NumOfTrain)
	
	def Error(self, t, a):		
		return sum(t*np.log(a))
		

	def Train(self, args):
		self.Initialization(args)
		
		n = 0
		while(n < self.NumOfEpoch):
			error, class_error = self.BackProp(args.filename[0])
			print "training error", error, class_error
			self.train_err.append(error)
			self.train_class_err.append(class_error)
			error_v, class_error_v = self.Valid(args.filename[1])
			print "validation error", error_v, class_error_v
			self.valid_err.append(error_v)
			self.valid_class_err.append(class_error_v)
			n += 1
		print "training finished!"
		
		
	def Valid(self, valid_file_name):
	
		error = 0
		true_label = 0
		for l in open(valid_file_name).read().splitlines():

			## read the txt file: target, input(784)
			line = l.split(',')
			target = int(line[-1])
			del line[-1]
			#inputs = np.append([1], np.array(map(float, line)))
			inputs = np.array(map(float, line))

			## compute output for hidden layer and store it in a1
			z1 = np.dot(self.w1, inputs)+ self.b1
			a1 = self.Sigmoid(z1)
			
			## compute output for output layer and store it in a2
			z2 = np.dot(self.w2, a1) + self.b2
			
			## compute softmax
			a2 = self.Softmax(z2)
			
			if (a2.argmax()==target):
				true_label += 1
			
			## now doing backprop, hidden -> output
			## first comput target 
			t = np.zeros(self.NumOfOutput)
			t[target] = 1
			#t = self.Softmax(t)

			error += -self.Error(t, a2)
		
		return error/self.NumOfValid, 1-true_label/float(self.NumOfValid)
			
		
	def Plot(self):
		t = np.arange(0, self.NumOfEpoch, 1)
		plt.plot(t, self.train_err, 'r--', t, self.valid_err, 'b--')
		plt.show()
		plt.plot(t, self.train_class_err, 'r--', t, self.valid_class_err, 'b--')
		plt.show()
		
	def PlotWeight(self):
	
		plotimg = np.zeros([280, 280])
		for x in range(plotimg.shape[0]):
			for y in range(plotimg.shape[1]):
				plotimg[x][y] = self.w1[10*(x/28) + y/28][28*(x%28) + y%28]
				
		plt.imshow(plotimg, cmap='gray')
		plt.show()
				


if __name__ == "__main__":
	start_time = time.time()
	parser = argparse.ArgumentParser(description='script for testing')
	parser.add_argument('filename', nargs='+')
	parser.add_argument('--dropout', type=float, defualt=0.5, help='the dropout vallues')
	parser.add_argument('--rate', type=float, default=0.1, help='The learning rate')
	parser.add_argument('--epoch', type=int, default=200, help='the number of epoch')
	parser.add_argument('--momentum', '-m', type=float, default=0, help='momentum parameter')
	parser.add_argument('--hidden', type=int, nargs='+', default = 100, help='the number of hidden units for each layer')
	args = parser.parse_args()
	NN = NeuralNetwork()
	NN.Train(args)
	NN.Plot()
	NN.PlotWeight()
	print("--- %s seconds ---" % (time.time() - start_time))

	
