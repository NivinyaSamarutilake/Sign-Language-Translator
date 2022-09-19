# Sign-Language-Translator
Semester 5 project for CS3282

## Overview
Communication between a speech impaired person and a normal person can be difficult if the latter does not have enough knowledge about sign language. Most of the general population lacks sign language communication skills. A third party is needed for the translation from sign language to verbal language in these situations. Therefore, it is important to develop a mechanism for direct communication between the deaf/mute community and the general population.

The sign language translator app is a solution to the above problem scenario. It is a real-time translator app that uses an image-classification model to differentiate and identify the various hand signs. The images will be captured by the smartphone, and they will be fed into a trained model which will output the corresponding letter as text. This app will translate the ASL alphabet into English. 

![image](https://user-images.githubusercontent.com/91209506/191010582-1e95f265-823b-4ea8-b15f-2acb8285a44e.png)


## Development

The [dataset]( https://www.kaggle.com/datasets/grassknoted/asl-alphabet) that was selected for this project has 87,000 images. The training set contains images for all letters of ASL alphabet and for space character. This dataset is quite commonly used in many of the previous research works. 

For the development of the app Android studio will be used and to run inference, Tensorflow Lite will be used. Tensorflow Lite has been optimized for on-device ML by addressing key constraints such as latency, connectivity (no need of internet connection), privacy of data, size of the model and power consumption. This will enable the data to be processed locally instead of being sent to a remote server, which will reduce latency as a result.

A CNN model will be used for this project. Therefore, feature extraction is implemented here.

## Evaluation of the model  

To evaluate, 30% of the above data will be taken as the test set. 

## Proposed Timeline

![image](https://user-images.githubusercontent.com/91209506/190921145-8eedefb2-942a-4195-b7a5-5734f69094fc.png)
