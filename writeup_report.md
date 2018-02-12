# **Behavioral Cloning**



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
[image4]: ./examples/flip-center.png "Flip Image"
[image5]: ./examples/nvidia-arch.png "NVidia Architecture"
[image6]: ./examples/loss-mse.png "Loss Plot"
[image7]: ./examples/steering-angle.png "Steering Line Plot"
[video1]: ./video.mp4 "Run Video"

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
* model.h5 containing a trained convolution neural network
* writeup_report.md for summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
I downloaded and ran the simulator on an iMac set for maximum computational efficiency with screen resoluton of 640x480 with the fastest graphics quality. My computer uses a trackpad, so I controlled the car with just the keyboard inputs.
I trained the model by recording three main runs with the car.
* One run with the car driving right through the middle of the road in the CCW direction for a one full lap.
* Another run with one lap in the CCW direction and another in the CW direction around the track.
* Another recording of the problematic part of the track just before the bridge and the bridge itself.
* I modified drive.py to make sure the inputs image fed to the network is always in RGB mode by using the BGR2RGB conversion function in cv2 module.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
Please note that I did not find any guidelines on building this model.py file, so I simply cut and pasted the bare model build in my notebook. To run this code on a dataset, the directory and file variables must be filled in corectly at line 37 in model.py. I will be glad to revise this if needed.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model is a copy of the well known NVidia Sequential network. The input color image of size 160x320x3 is cropped into a region of interest of only 90x320x3 such that only the relevant road in front of the car is worked on (model.py line 17). The image is then normalized by the Keras lambda layer which converts the pixel values of the image to lie between -0.5 and 0.5 (model.py line 21). The normalized image is then fed into a convolutional neural network which consists of three 5x5 filters and two 3x3 filter sizes (model.py lines 25-44). I also put in MaxPooling2D layers the first and third convolution outputs. The final 3x3 convolution layer outputs are Flatten ed  out into a Dense network with 1164 outputs (model.py lines 48). This is followed by 2 more Dense layers with 100 and 50 outputs (model.py lines 56-79).  (model.py lines 25-44)
I added the RELU layer to the outputs of all the dense layers to introduce nonlinearity. (code line 56, 65 and 74).

#### 2. Attempts to reduce overfitting in the model

I also put in three Dropout layers to make the network more robust to missing data. (model.py lines 51, 61 and 70).

The model was trained and validated on three data sets from 3 runs in the simulator to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).
Even though the adam optimizer does not need any parameter to be set, I tried to improve the model by reloading the best current model.h5 and then fitting with new data from runs which emphasized driving in problematic areas such as the road before the bridge and after it. For this purpose, I had to lower the learning rate from its default of 0.001 to 0.0001 to keep the good weights from the old model and update with the good data in the new data set.
The other major parameters I tried to tune were the keep_prob value for the Dropout layers. I finally decided on 0.5 for this based on some experiments and watching the loss plots.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. I made training sets for driving in the CCW direction and also the CW direction in the track. I also separately recorded data for recovering from the left and right sides of the road for problematic parts of the track such as the section before the bridge and after that.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVidia car driving system shown in the figure below. I thought this model might be appropriate because it has the almost identical set of inputs of 3 cameras looking forward set at the center, left and right of the car. The output also is identical to our case which generates the steering angle from the input center camera image.
![alt text][image5]
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set in a 80:20 ratio. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include dropout layers with a keep_prob=0.5. This allowed the network to train with randomly missing data.

Then I also added the ELU layers to add non linearity to the system to make it even more robust.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as the part of the track just before the bridge and then the curved partly marked road past the bridge. to improve the driving behavior in these cases, I recorded driving in a recovery mode (not centerline driving) through these stretches and fed them into the model.
In this case, I set a variable called make_model to False. This allows for the code to read in the current best model called 'model-mydata-best.h5' using the keras load_model function. This allows me to keep the old working model and gradually improve it with the new dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road in most cases.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes
* Lambda Layer: This layer normalizes all the pixels in the input image to a value between -0.5 to 0.5.
* Three 5x5 Convolution layers followed by MaxPooling2D to allow the neural network to learn and extract the road image features that assist in keeping the car in the middle of the road.
* Two 3x3 Convolution layers following the output of the 5x5 convolution layers.
* A Flatten layer which takes the output 2D image from the previous convolution layer into a 1D vector.
* Dropout layer with a keep_prob of 0.5 to allow the network to handle loss of data.
* Four Dense layers along with exponential linear unit ELU.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example images of center lane driving. The three images correspond to the left center and right images.

![alt text][image1]

One of the crucial steps in the preprocessing of the image data was to crop the image to just the region of interest as far as road following is considered. After experimentation, I found that cropping the incoming images from its 160x320x3 to 90x320x3. The figures below show the cropped left center and right images.

![alt text][image2]

One important factors when training with driving data is that the number of images corresponding to near zero steering is an order of magnitude larger than the ones where there is a significant steering value. This is due to the fact that the straight sectors of the track are usually statistically longer than the stretches with curves. To make sure the training data has sufficient steering input, I artificially reduced the zero steer images by randomly discarding 90% of them. On the other hand I increased the contribution of images where the steering is significant by a factor of four.
* Using the center image and its associated steering angle.
* Using the left image and associating it with a steering angle + 0.25.
* Using the right image and associating it with a steering angle - 0.25.
* Flipped the center image and associated it with a negative steering angle.

For the 10% of zero steer images that pass, I translated the image by a random uniform distribution of 100 pixels in the X axis. The center image is also transformed based on the translation and the steering angle is also augmented by the random translation.
This technique allowed the training data to train with images with significant steering angles.
The center image and its translated image with a random translation is shown below.

![alt text][image3]

The center image and its flipped image is shown below.

![alt text][image4]
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. This data is in mydata4.

I separately recorded a drive on track two. Surprisingly, I found that the model.h5 it generated was able to drive autonmously for a fair distance along the path despite being a radically different environment. This demonstrates that the model is robust. I hope to augment the model trained on the first track with the data from the second track if that is possible in the coming days.

#### 3. Loss vs epochs

The model mean square error loss vs epochs is shown below. I used only 3 epochs for training. I revised the model architecture to reduce overfitting the data. I used Dropouts and ELUs and MaxPooling to reduce overfitting.
![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would help improve the driving in a CCW track with the training data from a CW track run. For example, here is an image that has then been flipped:

![alt text][image4]

I also obtained the model independently from the data that was supplied by udacity. This confirmed that the architecture was valid for data that was not limited to the data I generated with my simulator.

I also plotted the steering for a training dataset with just 3 images which showed a zero steer and full positive steer and a full negative steer. After the model was saved. I used model.predict() function to confirm that it was doing the right thing with the basic images.

![alt text][image7]

The movie I made of successfully running the simulator for a whole lap is in the video.mp4 in this directory.

![alt text][movie1]

For a typical run there was 2542 images. This was split 80:20 to 2033 training images and 509 for validation.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was found to be 3. I found this by experimenting with different datasets and number of epochs which showed a steady monotonic drop in the loss with epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary except when merging models from different datasets.

#### 3. Stuff to do
Note that I did not use the generator as I could not figure out an elegant way to incorporate the random 10% probability for zero steer images into a fixed sample size set. I would like to do this in the coming days if I can revise this project. I did make a generator but it looked like the model performed worse in my experiments than when made without it.

#### 3. References
My mentor provided me with several links to help with this project. I also looked at the notes provided in the forums and suggested by the course.
