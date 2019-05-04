# **Behavioral Cloning Project** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/driving.png "Driving Image"
[image2]: ./examples/center.jpg "Normal Image"
[image3]: ./examples/center_flipped1.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 50-71) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 51-54). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 73).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried to control the car frequentry to keep the car on the center when I recorded the data. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one Nvidia used in [End to end learning for self-driving cars](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwjyk96Z_IDiAhVLzbwKHewNBwEQFjAAegQIAhAC&url=https%3A%2F%2Fimages.nvidia.com%2Fcontent%2Ftegra%2Fautomotive%2Fimages%2F2016%2Fsolutions%2Fpdf%2Fend-to-end-dl-using-px.pdf&usg=AOvVaw10_flEW7gmCuHMDUngG8qV) I thought this model might be appropriate because the task I'm trying to do is similar to the Nvidia's work.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduced dropout layers to Convolutional layers and fully connected layers.

Before adding dropout layers  
 Epoch 1/5 loss: 0.0386 - val_loss: 0.0525  
 Epoch 2/5 loss: 0.0316 - val_loss: 0.0485  
 Epoch 3/5 loss: 0.0266 - val_loss: 0.0519  
 Epoch 4/5 loss: 0.0230 - val_loss: 0.0565  
 Epoch 5/5 loss: 0.0198 - val_loss: 0.0646  
 
After adding dropout layers  
 Epoch 1/5 loss: 0.0411 - val_loss: 0.0517  
 Epoch 2/5 loss: 0.0348 - val_loss: 0.0504  
 Epoch 3/5 loss: 0.0311 - val_loss: 0.0521  
 Epoch 4/5 loss: 0.0286 - val_loss: 0.0517  
 Epoch 5/5 loss: 0.0265 - val_loss: 0.0516  

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

|Layer (type)                | Output Shape            |  Param #|   
|:---|:---:|---:|
|lambda_1 (Lambda)            |(None, 160, 320, 3)       |0         
|cropping2d_1 (Cropping2D)    |(None, 65, 320, 3)        |0         
|conv2d_1 (Conv2D)            |(None, 31, 158, 24)       |1824      
|conv2d_2 (Conv2D)            |(None, 14, 77, 36)        |21636     
|conv2d_3 (Conv2D)            |(None, 5, 37, 48)         |43248     
|conv2d_4 (Conv2D)            |(None, 3, 35, 64)         |27712     
|dropout_1 (Dropout)          |(None, 3, 35, 64)         |0         
|conv2d_5 (Conv2D)            |(None, 1, 33, 64)         |36928     
|dropout_2 (Dropout)          |(None, 1, 33, 64)         |0         
|flatten_1 (Flatten)          |(None, 2112)              |0         
|dense_1 (Dense)              |(None, 100)               |211300    
|dropout_3 (Dropout)          |(None, 100)               |0         
|dense_2 (Dense)              |(None, 50)                |5050      
|dense_3 (Dense)              |(None, 10)                |510       
|dense_4 (Dense)              |(None, 1)                 |11        


Total params: 348,219  
Trainable params: 348,219  



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

To augment the data sat, I also flipped images since the most of the corners are left-turn and the model will tend to turn left. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]

I also used images from left and right camera. This will help the model to learn more different data. I shifted the value of the steering angle for each images 0.2 deg.


After the collection process, I had 30870 number of data points. I then preprocessed this data by cropping the image


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was choosed as 3 since the validation loss doesn't improve or get worse (overfitting) after 3-4 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Result
You can see the test result in video.mp4. The car doesn't get off from the road and navigate the road correctly.






