# Intro-to-AI
Acknowledgement: This project is based on the one created by Dan Klein and John DeNero that was given as part of the programming assignments of Berkeley’s CS188 course. 
In this project, you will design three classifiers: a naive Bayes classifier, a perceptron classifier and a classifier of your choice. You will test your classifiers on two image 
data sets: a set of scanned handwritten digit images and a set of face images in which edges have already been detected. Even with simple features, your classifiers will be able 
to do quite well on these tasks when given enough training data. Optical character recognition (OCR) is the task of extracting text from image sources. The first data set on 
which you will run your classifiers is a collection of handwritten numerical digits (0- 9). This is a very commercially useful technology, similar to the technique used by the
US post office to route mail by zip codes. There are systems that can perform with over 99% classification accuracy (see LeNet-5 for an example system in action). Face detection
is the task of localizing faces within video or still images. The faces can be at any location and vary in size. There are many applications for face detection, including human 
computer interaction and surveillance. You will attempt a simplified face detection task in 1 CS440: Intro to AI which your system is presented with an image that has been 
pre-processed by an edge detection algorithm. The task is to determine whether the edge image is a face or not. Please refer to http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/ classification.html for a brief description of the Perceptron and Naive Bayes classifiers.


What you should do:
1. Implement three classification algorithms for detecting faces and classifying digits:
(a) Perceptron
(b) Naive Bayes Classifier
(c) Any other algorithm of your choice (Neural Networks, Decision Trees, or any other
you are interested in).
2. Design the features for each of the three problems, and write a program for extracting the
features from each image.
3. Train the three algorithms on the part of the data set that is reserved for training. First, use
only 10% of the data points that are reserved for training, then 20%, 30%, 40%, 50%, 60%,
70%, 80%, 90%, and finally 100%. All the results should a function of the number of data
points used for training.
4. Compare the performances of the three algorithms using the part of the data set that is
reserved for testing, and report:
• The time needed for training as a function of the number of data points used for
training.
• The prediction error (and standard deviation) as a function of the number of data
points used for training.
5. Write a report describing the implemented algorithms and discussing the results and the
learned lessons.
