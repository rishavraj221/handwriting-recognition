# handwriting-recognition

This is a project where a pen fitted with IMU is trained to recognize handwritten numerals (0 to 9) by Naveen, daytime bioinformatician, maker at heart...

My basic purpose is to analyze which, why and how things are used in this project along with the code...

Hardware components:
* SparkFun RedBoard Artemis ATP	
* SparkFun 9DoF IMU Breakout - ICM-20948 (Qwiic)
* SparkFun Qwiic Cable - 500mm
* Grove - Mech Keycap	
* Seeed Grove - Mech Keycap
* Gameduino 3

Software apps and online services:
* TensorFlow	
* Jupyter Notebook	
* Arduino IDE	

Hand tools and fabrication machines:
* Soldering iron (generic)	
* Solder Wire, Lead Free

Overview

In this project, I build a pen device which can be used to recognize handwritten numerals. As its input, it takes multidimensional accelerometer and gyroscope sensor data. Its output will be a simple classification that notifies us if one of several classes of movements, in this case 0 to 9 digit, has recently occurred.

Data collection for training

The first and the most important step in a machine learning project is to collect the training data in such a way that it should cover most of the representative cases for a given classification task. To capture accelerometer and gyroscope data in a discrete real-time steps is a time consuming and error-prone task. I spent most of the time to collect data and look at it if it was captured correctly. To overcome this tedious and repetitive task I developed a user interface. I wanted to make the user interface portable and easy to use. I used Artemis ATP development board with a Gameduino 3 touchscreen (an Arduino shield) to present a user interface which allows to select numeral (0 to 9) and also the last readings can be deleted if there was some error while data capturing. A SparkFun 9DoF IMU Breakout - ICM-20948 (Qwiic) is used to capture the accelerometer and gyroscope data. The IMU Breakout is attached to a pen close to the tip and it is connected to the Artemis ATP using a long (50cm) Qwiic cable.

To capture time-series data from the IMU during the pen movement should be quick otherwise there can be unwanted noise at the beginning and the end. Using the touchscreen to start and stop the capturing was a bit slow since the screen needs right amount of pressure to response. To circumvent this issue I used a mechanical switch which is very sensitive to the clicks and did the right job. Since Gameduino 3 shield covers all the power pins of Artemis ATP, I had to use secondary rail of plated through-holes on the Artemis ATP to solder wires to connect to the mechanical switch. Thanks to the ATP (All the pins)!

The captured data is saved to the files on a micro SD card attached to the Gameduino 3. Each pen movement data was captured as a separate file. The file contains no header line, only the multiple lines of the comma separated accelerometer (3-axis) and gyroscope (3-axis) data in a format as accel_X, accel_Y, accel_Z, gyro_X, gyro_Y, gyroZ. An example is given below.

Data splits and augmentation

A little over 100 samples for each digits (0-9) were captured. The collected data has been split into training (60%), validation (20%), and testing (20%) datasets. Since the data was collected from the IMU sensor using scaled (16 bit) and digital low pass filter (DLPF) setting and they are already within a specified range of the accelerometer and gyroscope readings so we can use the raw data as is for training and inferencing. The captured training dataset is still small for a deep learning model so each sequence of readings were modified slightly from the original. The modifications include shifting and time warping the samples, adding random noise, and increasing the amount of acceleration. This augmented data is used alongside the original data to train the model, helping make the most of our small dataset. The training data was padded to keep only 36 sequences of the readings.

Model Architecture

Each input data has 36 sequences of 6 readings (3-axis accelerometer, 3-axis gyroscope). We can think of the input data as an image of 36x6 pixels. A convolutional neural network is one of the best options suited for recognizing patterns in images and time-series sequence data. In our case it is time-series motion data. The first few layers are 2D convolution neural networks with few other regularization layers. The last layer is a fully connected dense layer with softmax activation which outputs a probability of all 10 classes. The summary of the model is given below.

Model Training and Evaluation

The training of the model was done on an Intel NUC with Linux and an eGPU (NVIDIA GTX 1080Ti). Although it just takes couples of minutes to train on a CPU but the development process becomes pretty slow while testing out different architectures and hyper-parameters. The TensorFlow 2.1 with Keras API is used for model creation and training process. I created a Jupyter notebook for data processing, training and the final model conversion. All code are available at Github repository which is mentioned in the code section. The training accuracy is 94.8% and evaluation accuracy on test data is 93.3% which can be further improved with more training datasets and model hyper-parameters tuning.

Inferencing on the device

The created model is converted to the TensorFlow Lite model and the converted model is transformed into a C array file for deploying with the inferencing code. The TensorFlow Lite Micro SDK is used to run inference on the device. I have created an Arduino sketch (handwriting_recognizer.ino available at Github repository) for inferencing and displaying the result. The setup is the same as the training setup except we do not use mechanical switch. The Artemis ATP receives the samples continuously from the IMU sensor and outputs the highest probability class on the Gameduino 3 shield display. The inferencing rate depends on the device and the model size. In our case model is a bit large due to many layers and the inferencing rate is 3 to 4 inferences per second. Since the microcontroller is busy during inference and we do not want to lose any IMU samples during that period. To overcome this issue FIFO is used in IMU module. When I started using SparkFun 9DoF IMU Breakout - ICM-20948 (Qwiic) there was no FIFO implementation in the library. I forked the Sparkfun library Github repository and implemented the FIFO. The forked can be found here: https://github.com/metanav/SparkFun_ICM-20948_ArduinoLibrary.

Inferencing demo

The live demo is below. It is not perfect but it works.

https://youtu.be/r3p7ybnA3TU

The use cases for common benefit
It is an easy to use low-powered device which can run on a coin cell battery for weeks (without display). It can be used safely for kids who are learning to write alphabets (needs training data) and numbers with the help of LEDs for result output.

Thank you
