import sys
import numpy as np
import math
import matplotlib.pyplot as plt

class NeuralNetwork():

	def FormatData(self, filename):

		for l in open(filename).read().splitlines():
			line = l.split(',')
			label = line[-1]
			#del line[-1]
			self.PlotData(line)
			


	def PlotData(self, pixel_array):
		data = np.zeros((28,28))
		for (x,y), value in np.ndenumerate(data):
				data[x,y] = float(pixel_array[28*y+x])	
		plt.imshow(data, interpolation='nearest')
		plt.show()


if __name__ == "__main__":

	NN = NeuralNetwork()

	NN.FormatData(sys.argv[1])