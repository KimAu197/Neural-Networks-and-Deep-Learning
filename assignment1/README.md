# <center> Project 1 </center>

## 1 Neural Network
In this problem we will investigate handwritten digit classiﬁcation. The inputs are 16 by 16 grayscale images
of handwritten digits (0 through 9), and the goal is to predict the number of the given the image. If you run
example_neuralNetwork it will load this dataset and train a neural network with stochastic gradient descent,
printing the validation set error as it goes. To handle the 10 classes, there are 10 output units (using a {−1, 1}
encoding of each of the ten labels) and the squared error is used during training. Your task in this question is to
modify this training procedure architecture to optimize performance.
Report the best test error you are able to achieve on this dataset, and report the modiﬁcations you made to
achieve this. Please refer to previous instruction of writing the report.
## 1.1 Hint
Below are additional features you could try to incorporate into your neural network to improve performance (the
options are approximately in order of increasing diﬃculty). You do not have to implement all of these from the
scratch, and we provide some demo codes. The modiﬁcations you make in trying to improve performance are up to
you and you can even try things that are not on the question list in Sec. 1.2. But, let’s stick with neural networks
models and only use one neural network (no ensembles).

老师提供了十个问题，有相关的matlab代码以及Project题目在original文件夹里，code文件夹里有剩余10个问题的代码