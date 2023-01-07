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







