# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[barchart]: ./writeup_images/barchart.png "Visualization"
[before_greyscale]: ./writeup_images/before_greyscale.png "before greyscale"
[after_greyscale]: ./writeup_images/after_greyscale.png "after greyscale"
[barchart_augmented]: ./writeup_images/barchart_augmented.png "Visualization"
[original]: ./writeup_images/before_greyscale.png "original"
[rotated]: ./writeup_images/rotated_image.png "rotated"
[translated]: ./writeup_images/translated.png "translated"
[scaled]: ./writeup_images/scaled.png "scaled"

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/00068.png "Traffic Sign 1"
[image5]: ./test_images/00069.png "Traffic Sign 2"
[image6]: ./test_images/00070.png "Traffic Sign 3"
[image7]: ./test_images/00071.png "Traffic Sign 4"
[image8]: ./test_images/00072.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

It is a bar chart showing how many sign we have in the dataset for each classes

![alt text][barchart]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale as suggested in the lenet paper, and it 
reduced the processing time

Here is an example of a traffic sign image before and after grayscaling.

![alt text][before_greyscale]
![alt text][after_greyscale]

I decided to generate additional data because some of the classes seemed to have very few data compared to the others. So i used a combination of shifting images, rotation and scaling.
the goal was to have at least 1000 images per classes.
As a last step, I normalized the image data to improve the performances of the neural network
Here is an example of an original image and an augmented image:

![alt text][original]
![alt text][rotated]
![alt text][scaled]
![alt text][translated]

this is the histogram of the data classes with the added data 
![alt text][barchart_augmented]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 greyscale image   					| 
| Convolution 4x4     	| 1x1 stride                                    |
| RELU					|												|
| Max pooling	      	| 2x2 stride				                    |
| Convolution 4x4	    | 1x1 stride                                    |
| RELU					|												|
| Max pooling	      	| 2x2 stride 				                    |
| Convolution 4x4	    | 1x1 stride                                    |
| RELU					|												|
| Max pooling	      	| 2x2 stride				                    |
| Concatenate           | Concatenate results from layer1 and layer3    |
| dropout               | drop out with a probability of 0.5            |
| Fully connected		|       									    |
| Softmax				|                   							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

i used an Adamoptimizer
epoch 10 
batch size 256

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.949
* test set accuracy of .924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used the lenet architecure from the lenet lab project because it worked for image recognitions of handwritten numbers
* What were some problems with the initial architecture?
The accuraccy of this model was 90% at best
Nothing prevented it from overfitting


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

- I introduced a dropout to prevent overfitting
- I also follow the connected the result of the first layer to the last layer as suggested in the lenet paper
- I modified changed the size of the first layer of convolution to use a 4x4 matrix. This value was found by trial and error

* Which parameters were tuned? How were they adjusted and why?
the dropout ratio
the size of the matrix of the first convolution. reducing the size to 4x4 yielded the best results

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
- The convolution layer allows the algoritm to accentuate the features of an image to facilitate the recognition
- the dropout is used to prevent overfitting on the training model
- connecting the result of the first layer to the last allows the network to base its decision on both higher level and low level features of the image 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The main difficulty might come from the fact that those image seem to have a very low resolution.
and image 2 and 3 are very under exposed which makes teh features on them harder to distinguish


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing     | Children Crossing   							| 
| 80 km/h     			| 80 km/h										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 36th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were


image 1
| Probability       |     Prediction	               | 
|:-----------------:|:--------------------------------:| 
| 0.982             | Children crossing                |
| 0.010             | Bicycles crossing                |
| 0.005             | Beware of ice/snow               |
| 0.003             | Road work                        |
| 0.000             | Road narrows on the right        |


image 2
| Probability       |     Prediction	               | 
|:-----------------:|:--------------------------------:| 
|0.913              | Speed limit (80km/h)             |
|0.085              | Speed limit (30km/h)             |
|0.001              | Speed limit (50km/h)             |
|0.000              | Speed limit (20km/h)             |
|0.000              | Speed limit (70km/h)             |

image 3
| Probability       |     Prediction	                     | 
|:-----------------:|:--------------------------------------:| 
| 0.931             | Beware of ice/snow                     |
| 0.039             | Right-of-way at the next intersection  |
| 0.018             | Road narrows on the right              |
| 0.003             | Children crossing                      |
| 0.003             | Dangerous curve to the right           |

image 4
| Probability       |     Prediction	               | 
|:-----------------:|:--------------------------------:| 
| 1.000             | Road work                        |
| 0.000             | Wild animals crossing            |
| 0.000             | Keep right                       |
| 0.000             | Road narrows on the right        |
| 0.000             | Beware of ice/snow               |

image 5
| Probability       |     Prediction	                    | 
|:-----------------:|:-------------------------------------:| 
| 1.000             | Right-of-way at the next intersection |
| 0.000             | Beware of ice/snow                    |
| 0.000             | Double curve                          | 
| 0.000             | Road narrows on the right             |
| 0.000             | Pedestrians                           |


image 1
| Probability       |     Prediction	        					| 
|:-----------------:|:---------------------------------------------:| 
|  0.989            | Children crossing                 
|  0.011            | Bicycles crossing                 
|  0.000            | Beware of ice/snow                
|  0.000            | Slippery road                     
|  0.000            | Road narrows on the right         


image 2
| Probability       |     Prediction	        					| 
|:-----------------:|:---------------------------------------------:| 
| 0.998             |  Speed limit (80km/h)                         |
| 0.001             |  Speed limit (100km/h)                        |
| 0.001             |  Speed limit (70km/h)                         |
| 0.000             |  Speed limit (20km/h)                         |
| 0.000             |  Speed limit (30km/h)                         |





For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


