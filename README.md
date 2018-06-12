
# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the numpy library to calculate length and shape of traffic signs data set. The data in question is the pickled dump of the images with training, testing and validation images in the numeric (pixel-value) form.

* The size of training set is : 34799
* The size of the validation set is : 4410
* The size of test set is : 12630
* The shape of a traffic sign image is : 32 x 32 x 3
* The number of unique classes/labels in the data set is : 43

#### 2. Visualization of the dataset.

Available in the html of the python notebook. It is just plotting the images with the sign-names that are associated with it.

### Design and Test a Model Architecture

#### 1. Preprocessing the image data. 
- The images were first changed to grayscale THis was done because the paper mentioned the accuracy of the training model won't increase by keeping color and grayscale is easy/faster to process.http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
- After converting to grayscale the images were scaled from [0, 255] to [0,1].
- The shape of the image was increased from 3 to 4 to fit into the tensorflow model.

#### 2. Model architecture

I used the Lenet Transformation code to whip out a smaller more reusable code to create two functions 1 for convolution and 1 for connected layer.

I then created a convolution neural network by calling Convolution twice followed by a connected layer. The convolution has relu followed by a maxpool. The first connected layer has relu and the second doesn't.  

| 1st Layer convolution      | 2nd Layer convolution      | 3rd Layer Fully Connected  | 4th Layer Fully Connected| 
|:--------------------------:|:--------------------------:|:--------------------------:|:-------------------------|
| Input = 32x32x3x1 RGB image| Input = 32x32x3x1 RGB image|                            |    					  |
| Filter = 5x5               | Filter = 5x5               |                            |    					  | 
| Depth = 64                 | Depth = 32                 |    						   |                          |
| Stride = 1x1               | Stride = 1x1               |                            |                          |
| Padding = same             | Padding = same             |                            |                          |
| RELU = True				 | RELU = True				  | RELU = True				   | RELU = False             |
| No Dropout				 | No Dropout				  | Dropout Probability = 0.8  | Dropout Probability = 0.8|
| Max pooling = 2x2      	 | Max pooling = 2x2      	  |                            |                          |
| Output = 32x32x64			 | Output = 16x16x32		  | Output = 256	           | Output = 43	          |


1. Get Logits from network. 
2. Calculate the probability of the logits using softmax function.
3. Calculate the Cross Entropy of the probability using the cross entropy function.
4. Calculate the mean for the loss operation of cross entropy.
5. Minimize the loss with the learning rate.
6. Setup Evaluation by getting Accuracy by comparing one hot encode label with out.

- The Parameters of the functions are mostly from the lecture itself. and they are all multiples of 16.
- Padding is same to keep the output of the same shape.
- The maxpool function is copied over from the Lenet transformation; it decreases the size of the output and 2x2 stride with same padding seemed good enough from the lectures.
- Dropouts is added in fully connected network for a value ranging from 0.8-1 to prevent overfitting.
- Relu functions are used for a non-linear system and remove the negative values.



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I re-used all the elements of the Lenet Transformation as suggested in the assignment. I used a Learning Rate o= 0.001, epoch= 25, batch size of 256. I wanted to use a lower learning rate but that was raising the value of epoch and that was taking testing longer. On increasing the learning rate to 0.001 decreased my accuracy but it was only 0.3 difference. I had a higher epoch but I couldn't afford that much time so I decreased it to 25. If I double it the accuracy will get better but I was happy with anything above 0.9.

The steps given above is how the network was trained almost everything is from the lectures like the adam optimizer, cross entropy softmax etc.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of : 0.933
* validation set accuracy of : 0933
* test set accuracy of : 0.928

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I copied over the Lenet Transformation from the lecture but it didn't allow for a lot of plug and play. So I broke it into two code-pieces (layer) 1 for convolution using conv2d and relu transformation and maxpool and the other into a fully connected layer and then ran the sequence of my neural network using those two functions.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I mostly changed the values of epoch and learning rate the rest is all Lenet transformation.
* Which parameters were tuned? How were they adjusted and why?
Almost all the paramters were tuned to increase speed and 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
All the weights and depths and biases are from Lenet transformation but the learning rate and epoch and batch size was changed to increase the speed and increase the accuracy.
If a well known architecture was chosen:
* What architecture was chosen?
I used the LeNet model.
* Why did you believe it would be relevant to the traffic sign application?
As it was suggested in the assignment itself I used LeNet to begin with. I was getting favorable reults in my first 10 epoch so I stuck to the design. I did not use any other architecture as this one worked out well for me. In hindsight I should have tried something that does some transformation on the images and the uses them for multi-layer processing to get a better accuracy.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The final accuracy is 0.933 which is quite high and with more epoch and lesser batch size we can get it higher. So, the model is defnitely working.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Can be seen in the accompanying pynb html.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| No Passing									| 
| Right Turn     		| Ahead Only 									|
| 60 Kmph Speed limit	| No Entry										|
| Pedestrian walk   	| General caution                               |
| Road Work			    | Road Work                              		|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. I am counting 3 to be correct because Pedestrian Walk is "General Caution" and Stop Sign means "No Passing". Stop should have said stop but "No passing" covered it.

Technically the accuracy is 20% as the road work sign worked.


