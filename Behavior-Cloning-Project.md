# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/orig-left-center-right.png "Original Images"
[image2]: ./examples/crop-left-center-right.png "Crop Images"
[image3]: ./examples/translate-center.png "Translate Image"
[image4]: ./examples/flip.png "Flip Image"
[image5]: ./examples/loss-mse.png "Loss Plot"
[image6]: ./examples/nvidia-architecture.png "NVidia Architecture"
[image7]: ./examples/train-valid-mse.png "Loss Plot2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality
**Code**

This link has my modified [project code](https://github.com/gvp-study/CarND-Behavioral-Cloning-P3.git)

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-good.h5 containing a trained convolution neural network
* Behavioral-Cloning-Project.md for summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
I downloaded and ran the simulator on an iMac set for maximum computational efficiency with screen resoluton of 640x480 with the fastest graphics quality. My computer uses a trackpad, so I controlled the car with the keyboard inputs.
I trained the model by recording three runs with the car.
* One run with the car driving right through the middle of the road.
* Another run with one lap in the CCW direction and another in the CW direction around the track.
* Another recording of the problematic part of the track just before the bridge and the bridge itself.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is a copy of the well known NVidia Sequential network. The input color image of size 160x320x3 is cropped into a region of interest of only 90x320x3 such that only the relevant road in front of the car is worked on (model.py line 17). The image is then normalized by the Keras lambda layer which converts the pixel values of the image to lie between -0.5 and 0.5 (model.py line 21). The normalized image is then fed into a convolutional neural network which consists of three 5x5 filters and two 3x3 filter sizes (model.py lines 25-44). I also put in MaxPooling2D layers the first and third convolution outputs. The final 3x3 convolution layer outputs are Flatten ed  out into a Dense network with 1164 outputs (model.py lines 48). This is followed by 2 more Dense layers with 100 and 50 outputs (model.py lines 56-79).  (model.py lines 25-44)
I added the RELU layer to the outputs of all the dense layers to introduce nonlinearity. (code line 56, 65 and 74).

#### 2. Attempts to reduce overfitting in the model

I also put in three Dropout layers to make the network more robust to missing data. (model.py lines 51, 61 and 70).

The model was trained and validated on three data sets from 3 runs in the simulator to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).
But I tried to improve the model by reloading the best current model.h5 and then fitting with new data from runs which emphasized driving in problematic areas such as the road before the bridge and after it. For this purpose, I had to tune the learning rate from its default of 0.001 to 0.0001 to keep the good weights from the old model and update with the good data in the new data set.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of th


For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVidia car driving system. I thought this model might be appropriate because it has the almost identical set of inputs of 3 cameras looking forward set at the center, left and right of the car. The output also is a classifier which generates the steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 80:20 ratio. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layers with a keep_prob=0.5. This allowed the network to train with randomly missing data.

Then I also added the ELU layers to add non linearity to the system to make it even more robust.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the part of the track just before the bridge and then the curved partly marked road past the bridge. to improve the driving behavior in these cases, I recorded driving in a recovery mode (not centerline driving) through these stretches and fed them into the model.
In this case, I set a variable called make_model to False. This allows for the code to read in the current best model called 'model-mydata-best.h5' using the keras load_model function. This allows me to keep the old working model and gradually improve it with the new dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road in most cases.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
