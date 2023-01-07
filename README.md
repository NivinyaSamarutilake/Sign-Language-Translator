# Sign-Language-Translator
Semester 5 project for CS3282

## Overview
Communication between a speech impaired person and a normal person can be difficult if the latter does not have enough knowledge about sign language. Most of the general population lacks sign language communication skills. A third party is needed for the translation from sign language to verbal language in these situations. Therefore, it is important to develop a mechanism for direct communication between the deaf/mute community and the general population.

The sign language translator app is a solution to the above problem scenario. It is a real-time translator app that uses an image-classification model to differentiate and identify the various hand signs. The images will be captured by the smartphone, and they will be fed into a trained model which will output the corresponding letter as text. This app will translate the ASL alphabet into English. 

![image](https://user-images.githubusercontent.com/91209506/191010582-1e95f265-823b-4ea8-b15f-2acb8285a44e.png)


## Development Plan

The [dataset]( https://www.kaggle.com/datasets/grassknoted/asl-alphabet) that was selected for this project has 87,000 images. The training set contains images for all letters of ASL alphabet and for space character. This dataset is quite commonly used in many of the previous research works. 

For the development of the app Android studio will be used and to run inference, Tensorflow Lite will be used. Tensorflow Lite has been optimized for on-device ML by addressing key constraints such as latency, connectivity (no need of internet connection), privacy of data, size of the model and power consumption. This will enable the data to be processed locally instead of being sent to a remote server, which will reduce latency as a result.

A CNN model will be used for this project. Therefore, feature extraction is implemented here.

### Evaluation of the model  

To evaluate, 30% of the above data will be taken as the test set. 

## Analysis of Previous Works

There are several works that have been done to address this issue. They have followed various machine learning techniques and here is the analysis on some of them.

1.  Convolutional Neural Networks (CNN) -
      * Most widely used technique in many of the previous works [2], [3], [4]
      * Very high accuracy - most have more than 95%
      * Suitable for systems with large datasets
      * Simple and has low computing times
2.  k-NN classifier - 
      * Less accuracy than CNN
      * Can be further developed to have more accuracy, but it will increase computation latency and complexity [1]
      * Not suitable for large datasets
3.  Support Vector Machines (SVM) -
      * Less accuracy compared to other models
      * Can be used in combination with other classifiers
  
Some of these works have combined 2 or more types of techniques to create an efficient model. 

## Proposed Timeline

Given below is a rough time plan for the project. Please note that it can be subject to change. 

![image](https://user-images.githubusercontent.com/91209506/191014363-42e6d0c4-4375-4482-838e-c2f2e865d767.png)

## References

[1] 	[Aryanie, D., & Heryadi, Y. (2015). American Sign Language-Based Finger-spelling Recognition using k-Nearest Neighbours Classifier. The 3rd International Conference on Information and Communication Technology. Bali.](https://www.researchgate.net/publication/279198249_American_Sign_Language-Based_Finger-spelling_Recognition_using_k-Nearest_Neighbours_Classifier)

[2] 	[Ojha, A., Pandey, A., & Maurya, S. (2020). Sign Language to Text and Speech Translation in Real Time using Convolutional Neural Network. International Journal of Engineering Research & Technology (IJERT).](https://www.ijert.org/research/sign-language-to-text-and-speech-translation-in-real-time-using-convolutional-neural-network-IJERTCONV8IS15042.pdf)

[3] 	[Tanseem N. Abu-Jamie, P. S.-N. (2022). Classification of Sign-Language Using Deep Learning by ResNet. International Journal of Academic Information Systems Research (IJAISR), 25-34.](https://philarchive.org/archive/ABUCOS-5)

[4] 	[Taskiran, M., Killioglu, M., & Kahraman, N. (2018). A Real-Time System For Recognition Of AmericanSign Language By Using Deep Learning. 41st International Conference on Telecommunications and Signal Processing. Athens.](https://www.researchgate.net/publication/326270945_A_Real-Time_System_For_Recognition_of_American_Sign_Language_by_Using_Deep_Learning)

## Implementation

The implementation of this project is mainly done in 2 parts. 
1. Development of the CNN model
2. Android app development

### Development of the CNN model

Here, it was decided to use a basic image classification CNN. I used Google Colab as the platform for training the CNN model.

#### Dataset

The dataset chosen for this project contains images for the 26 letters of the ASL alphabet, for the 'space', 'delete' signs, and images with no hand sign / hands shouwn in them to represent 'nothing'. Thus there are 29 classes in total and for each class there are 3000 images. The image given below shows the ASL alphabet hand signs.

![image](https://user-images.githubusercontent.com/91209506/211129744-13ccd1e5-dfdf-4ec9-9a08-57b3529c7f71.png)

A few of the main reasons why this particular dataset was chosen are,
* The number of images available for each class - For a machine learning / deep learning model, having a large number of data will ensure a better trained model with higher accuracy. This dataset offered a lot of data.
* For each class, there are images taken in different angles and with different lighting. Therefore, once trained, this model should be able to recognize hand signs done in different conditions with considereable accuracy, which is a key requirement in this project. 

Thie dataset was uploaded to MyDrive, so that it can be easily accessed in Google Colab.

As the first step, the dataset needed to be split into train, validation and test sets. 

Since this is a large dataset, a good proportion could be allocated for testing and validating purposes. Therefore it was decided to split the datset in the raio of *70 : 20 : 10* for *train : validation : test*. The exact numbers of the train, validation and test sets are given below.

| Set               | Total number of images  | Images per each class |
|-------------------|-------------------------|-----------------------|
|  Train            | 60900                   |  2100                 |
|  Validation       | 17400                   |  600                  |
|  Test             | 8700                    |  300                  |

The sklearn library offers a function to easily split a dataset according to the desired ratios, called [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) and this function was used here. 

```Python

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 12345)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.22222, random_state = 12345)

```

#### Preprocessing Data

The size of an image in this dataset is 200 x 200 pixels, and they are all in RGB format (jpg files). They cannot be processed in that format and had to be converted into a vector of some format - a numpy array or a pandas dataframe. 

Each pixel in an image has 3 values for R, G and B each. This can be visualized as follows.

<p  align="center">
     <img src="https://user-images.githubusercontent.com/91209506/211130586-78895f6e-ba67-44c0-a01d-e9e40cfa7207.png" width="200" height="200">
</p>

These RGB values can be read through the 'im_read()' function the in cv2 library. It returns a 3D numpy array with the shape (200, 200, 3) - representing the RGB values for each pixel along the height and width of the image. Once the array is flattened, for each image there are 200 x 200 x 3 = *12000* values in the range 0 - 255. 

For the training, values for all 60900 images, along with the image label (with the label, there would be 12001 values for each image) needed to be taken and reshaped into 1 large 2D matrix, with each row representing the 12001 values for the images, first column representing the labels, and the other columns representing the pixel values in the order R, G, B. The format of the matrix is shown below. 

<p  align="center">
     <img src="https://user-images.githubusercontent.com/91209506/211131047-cb0ae622-ee10-431a-87aa-7bad8ee5e779.png" width="300" height="200">
</p>

A problem was raised here, due to the large number of values. 

Google Colab has allocated 12 GB RAM and when the code was executed for this intensive task, it created a RAM overflow and the program crashed. As a solution, to reduce the program load, it was decided to resize the images to a much smaller size. Therefore, I resized the images to 32 x 32, making it 32 x 32 x 3 = 3072 values per image, and reducing the load by nearly 75%. The cv2 library also offered a 'resize' function for this purpose.

After performing this augmentation, the visualized data looked as follows:

<p  align="center">
     <img src="https://user-images.githubusercontent.com/91209506/211131779-e03e3047-bbe3-4823-b58c-b70ab0af66a4.png" width="300" height="300">
</p>

After obtaining the 2D training data matrix, the values were normalized by dividing it by 255. Now the dataset is preprocessed for training.

#### Training the CNN model

The 'Sequential' class Keras library was used to develop the CNN model. 3 layers were added with the ReLU activation function. Finally the model was compiled using the Adam optimizer. The summary of the CNN model is given below.

<p align="center">
     <img src="https://user-images.githubusercontent.com/91209506/211132341-708f8b2b-5191-4a19-afdb-382a7f5f9c93.png" width="400" height="500">
</p>

The dataset was then trained for 15 epochs and it managed to acheive high accuracy on the validation dataset. Then the test dataset was input to this CNN model and testing was carried out to see how much accuracy can be achieved with this model. The results are as follows.
* Training accuracy - 96.13%
* Validation accuracy - 99.46%
* Testing accuracy - 99.43%

The graph below visualizes the validation accuracy and validation loss of the model.
![image](https://user-images.githubusercontent.com/91209506/211132696-45ace1c9-8ba8-47fa-88a0-6814530e67c7.png)

#### Obtaining the .tflite model

Since the base CNN model acheived such high accuracy, it was decided to use the model as it is and integrate it to the android app. To perform this task, it was decided to use Tensorflow lite, as it is a deep learning framework that has been optimized for deploying machine learning models on mobile devices, microcontrollers and other such edge devices. Tensorflow lite can convert an already trained ML model to much more lightweight version in the .tflite format and this model can be embedded to the mobile app easily as it has been optimized for speed and storage considering that it is a key requirement in this project. The following few lines were just enough to perform this conversion.

```python

keras.models.save_model(cnn_model, 'model.pbtxt')
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model=cnn_model)
model_tflite = converter.convert()
open('signLanguageRecognitionModel.tflite', 'wb').write(model_tflite)

```
Now this tflite model should be integrated with the android app to perform further testing and get accurate translations of the hand signs.

### App development

My choice of platform to develop the sign language recognition app was Android Studio. It was chosen as it is easier to do testing of the mobile app real time, and it has solid tensorflow lite support tools. Another advantage is that Android Studio libraries and dependencies evolve faster along with the changing requirements of the mobile app development world. Thus there are many machine learning related tools that optimized the usage of ML models in Android Studio. With the newest developments it only requires a few lines of code to embed the ML models.

#### UI of the Sign Language Translator App

The UI of the app was designed to be easy to use. The blacked area of the interface is a PreviewView where it will display the preview of the footage we are recording. To record the video the camera access permission has to be granted first. Camera access only has to be given once when the app is starting up for the first time. With the next times of use, there is no need to give permission over again. 
The user only has to press the 'CAPTURE' button and hold the mobile and record the footage of the signer to get the images of the hand signs. The translation will appear in the designated area for the translation just below the camera surface. Once the user is done with the translation, he or she can press the 'STOP' button and the camera will stop recording. 

<img src="https://user-images.githubusercontent.com/91209506/211134073-1ddc3287-d6fb-421e-8ad0-db3db348f7ae.png" width="300" height="500" column_gap="40px"/>     <img src="https://user-images.githubusercontent.com/91209506/211134088-9439d37c-739a-421d-9c74-05c5c6172cc8.png" width="300" height="500"/>

#### Integrating the CNN model

The integration of the CNN model was easier to do with the Android Studio dependency for tensorflow lite. Then the .tflite model was imported as 'SignLangauageRecognitionModel.tflite' to the app codebase and placed in a new resource directory. A decision had to be made on how to input the images from the camera feed to the CNN model. There were 2 methods that I was able to come up with.
1. Set a timer at the moment the camera starts recording, and automatically take a photo at a certain trigger- for example, take an image every 1 second and input it to the model for processing
2. Extract frames from the live footage and input each of them to the model to run inference.

A major assumption that was made here was the signer's speed of doing the hand signs. It was assumed that the signer, at maximum, would be able to do about 1 handsign per second. This assumption is necessary in both methods to set a certain time interval between the translations, and I had to count for the delay that will be there for the processing of images.

Upon further research, I was able to find that the setting a trigger / alarm to take images and then inputting them to the model can be a tedious task compared to the 2nd method given above. The first method requires to save the images to a media storage as well and it can cause unnecessary delay and will take up storage.  With these findings, I was able to reject the first method completely and focus more on the 2nd method of extracting frames.

As mentioned above, Android Studio has a lot of fast updates and thus the API and documentations are constantly changing. The latest version of Camera API is [CameraX](https://developer.android.com/training/camerax), and it has a use case called 'Image Analysis' which can be used to extract frames and feed them into a machine learning model. This is a high performance analyzer with low latency, and it can extract frames up to 60fps according to the documentation. As this implementation aligned well with the real-time translation requirements of this project, I set up the Image Analysis use case for this project, and configured the image analyzer to send the frames from the live footage, process the frame, and input to the Sign Language Recognition model. 

A frame from the footage is defined by the 'ImageProxy' class, and once an ImageProxy object is taken, it is sent to the 'analyze()' method where the preprocessing of the input object happens. The analyzer is built using the image format as 'RGBA_8888' which is supported by CameraX, and the size of the frame is set to 200 x 200 similar to the training dataset used in the model.
First, the frame converted into a Bitmap image with a custom 'toBitmap()' method. Then the Bitmap is scaled down to 32 x 32 size as required by the model. Then it is converted to a Tensor Image which is then converted to ByteBuffer to perform the normalization (TODO). This ByteBuffer object is then loaded into a TensorBuffer of shape {1, 32, 32, 3} and this is the input to our Sign Language Recognition model. It will then generate the outputstream TensorBuffer. From this array, we should select the closest classification with the highest confidence percentage, and then output that class label as the classification.

#### Methodology 
With this method, the processing frame rate was calculated as nearly 30 fps. I have enabled a blocking mode for the analysis, thus, the other operations are blocked while a frame is being processed. Even with that, the app has acheived a high frame rate.

With the current implementation, there will be a lot of outputs just within the span of one second. This does not serve the purpose of the app as it is necessary to give a translation rate where the user can keep up with. Therefore, I decided to collect all the outputs of each set of 30 frames, and then find out the maximum prediction of them, and output it as the translation. This will give a translation approximately every 1 second. 

I am still on the process of figuring out if there is a better method to get the accurate translation.

#### Testing

Testing of the app is carried out through the emulator, both with virtual devices and the physical mobile device I am using. For the physical device I enabled developer options and executed the app through USB debugging. 

### Current Status of the project

So far the accuracy of the test results have been very low. I have identified some problem areas with the input frames, and plan on working on them. 
* Normalization of the inputstream values haven't been done
* The output keeps fluctuating between the same classes - A, C, O, Y 






