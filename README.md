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

