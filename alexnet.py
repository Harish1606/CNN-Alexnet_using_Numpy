from layer import operation

class Alexnet:
	def function(x):
		conv_output1=operation.convolution(x,no_of_filters=96,filter_size=11,stride=4,pad=0)
		pooling_layer1=operation.max_pooling(conv_output1,pooling_size=3,stride=2)
		
		conv_output2=operation.convolution(pooling_layer1,no_of_filters=256,filter_size=5,stride=1,pad=2)
		pooling_layer2=operation.max_pooling(conv_output2,pooling_size=3,stride=2)

		conv_output3=operation.convolution(pooling_layer2,no_of_filters=384,filter_size=3,stride=1,pad=1)
		conv_output4=operation.convolution(conv_output3,no_of_filters=384,filter_size=3,stride=1,pad=1)
		conv_output5=operation.convolution(conv_output4,no_of_filters=256,filter_size=3,stride=1,pad=1)

		pooling_layer3=operation.max_pooling(conv_output5,pooling_size=3,stride=2)
		flattening_output1=operation.flattening(pooling_layer3)

		fully_connected1=operation.forward_propagation(flattening_output1,hiddenlayer=4096,output=4096)
		fully_connected2=operation.forward_propagation(fully_connected1,hiddenlayer=2000,output=1000)
		Softmax=operation.softmax(fully_connected2)
		return Softmax 


		

