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
[image7]: ./testing_images.png

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
* The shape of a traffic sign image is 32x32x3
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

Learning rate = 0.00075

Batch size = 128, Number of Epochs = 20. The Epoch loop is terminated early if the desired accuracy is met.

Adam optimizer is used to train the model.

####4. Training approach

I started from the LeNet architecture, this gave an accuracy of 0.87.

I performed an iterative process where I started by increasing the training set size by using augmented images described earlier. Then, I increased the number of units in the network and also tuned the learning rate and number of epochs until I could not further increase the accuracy. To increase the units, I made the two convolutional layers deeper, and also added units to the fully connected layers. 

I repeated this process for multiple steps, each time increasing the number of training set images.

My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.932
* test set accuracy of 0.911


###Testing the Model on New Images

Here are five German traffic signs that I randomly selected from the testing data set.

![image7]

The model predicted the following:

For image sample 7707, predicted sign is  Testing Accuracy = 1.000
.. predicted sign is ...
['4', 'Speed limit (70km/h)']

For image sample 10045, predicted sign is  Testing Accuracy = 0.000
.. predicted sign is ...
['5', 'Speed limit (80km/h)']

For image sample 9642, predicted sign is  Testing Accuracy = 1.000
.. predicted sign is ...
['14', 'Stop']

For image sample 6837, predicted sign is  Testing Accuracy = 1.000
.. predicted sign is ...
['1', 'Speed limit (30km/h)']

For image sample 11466, predicted sign is  Testing Accuracy = 1.000
.. predicted sign is ...
['12', 'Priority road']

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

Discussion on difficulty in predicting an image:
The model fails to predict the second image correctly. Most likely this is because in the image, the speed limit sign is rotated by some angle. Thus although, the model correctly predicts it is a speed-limit sign, it incorrectly predicts it to be a an '80 km/h' sign, instead of 120 km/h sign. 
Other properties of image which can make it hard to predict are:
1. Bluriness
2. Not enough lighting or too much lighting (i.e. not enough contrast)

####3. Softmax probabilities

Below are the softmax probabilities that the model outputs for the above 5 testing images. The model is extremely confident about images 1,3,4,5. For image 2, which the model incorrectly predicts, it is not very confident.

TopKV2(values=array([[  1.00000000e+00,   8.75768902e-10,   8.27495851e-18,
          3.57787862e-20,   7.88174028e-22]], dtype=float32), indices=array([[4, 0, 1, 8, 7]], dtype=int32))
          
TopKV2(values=array([[ 0.56742024,  0.14815648,  0.087469  ,  0.07023293,  0.04088632]], dtype=float32), indices=array([[ 5, 40,  1,  2,  8]], dtype=int32))

TopKV2(values=array([[  1.00000000e+00,   3.53753881e-23,   1.39664742e-25,
          2.58132872e-28,   6.37091005e-29]], dtype=float32), indices=array([[14, 17,  4,  5, 13]], dtype=int32))
          
TopKV2(values=array([[  1.00000000e+00,   1.93647053e-20,   1.88926146e-20,
          3.09336879e-23,   9.83334840e-27]], dtype=float32), indices=array([[1, 5, 2, 0, 6]], dtype=int32))
          
TopKV2(values=array([[  1.00000000e+00,   5.43675660e-10,   9.60820312e-11,
          4.72825668e-11,   1.19804253e-11]], dtype=float32), indices=array([[12,  1, 17, 14, 15]], dtype=int32))




