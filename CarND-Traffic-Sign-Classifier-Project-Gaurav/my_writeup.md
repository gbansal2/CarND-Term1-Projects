#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./hist_nclasses.png "Histogram for n_classes"
[image2]: ./sample_images.png "Sample images"
[image3]: ./preprocessed_image.png
[image4]:  ./writeup_images/rot_imgt.png
[image5]: ./writeup_images/pers_imgt.png
[image6]: ./writeup_images/trans_imgt.png

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###1. Introduction

You're reading it! and here is a link to my [project code](https://github.com/gbansal2/CarND-Term1-Projects/blob/master/CarND-Traffic-Sign-Classifier-Project-Gaurav/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used basic python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data is divided among different classes. Note that some classes are much less represented than other classes. 

![Histogram of n_classes][image1]

Next, showing some randomly selected images:

![image2]

###Design and Test a Model Architecture

####1. Preprocessing of training data:

I applied the following pre-processing steps:

a. Converted the image to grayscale - this helps extract the most useful features of the image.

b. Centered the images around mean intensity values, and scaled them by standard deviation of the intensity values - this helps to make the min-max range of intensity values of different images comparable so that all the images are similarly weighted.

Here's an example of an image before and after preprocessing:

![image3]

To add more data to the the data set, I used the following geometric transformation on images. All of these were done using OpenCV functions.

a. Rotation by a random angle between -15 and 15 degrees. 

b. Perspective transformation.

c. Translation.

Here is an example of an original image and transformed image for the above mentioned transformations.

![image4]
![image5]
![image6]

For each transformation, the training data set is shuffled and then 20,000 transformed images are generated.  This process is repeated for the second and third transformations. The new data set contains 94799 images. 

####2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48    									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				|
| Flatten    | outputs 1200 |
| Fully connected		| outputs 240        									|
| RELU					|												|
| Fully connected		| outputs 168        									|
| RELU					|												|
| Fully connected		| outputs 43 (= n_classes)      									|

 
####3. Model training

I used the following methods and parameters for training the model:

Learning rate = 0.0008

Batch size = 128, Number of Epochs = 20. The Epoch loop is terminated early if the desired accuracy is met.

Adam optimizer is used to train the model.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

