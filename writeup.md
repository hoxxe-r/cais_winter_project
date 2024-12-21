# CAIS Winter Project
Aidan Parris ajparris@usc.edu

## 1- Project Presentation
I chose to do the facial expression recognition task. Because this dataset is images, I used a convolutional neural network that uses convolutional layers with 3x3 filters and batch normalization. There are three sections, each with increasing numbers of filters and with max-pooling layers after.

## 2- Dataset
This dataset contains seven expression classes: anger, disgust, fear, happy, sad, suprise, and neutral. There are 28,709 images in the train category and 7,178 images in the test category giving a train/test split of 0.8. The smallest category is disgust with a total of 547 photos and the largest category is happy with 8,989 images. Other than these two classes, the dataset is fairly balanced. The lack of examples of disgust led to issues achieving high accuracy for this class. Each image is 48 by 48 pixels and grayscale in the range \[0, 255].
![Img](/plots/class_count.png)

## 3- Model Development and Training
Initially, the model I had had a couple of convolutional layers that had ~1000 outputs when flattened then those fed into a neural network with dense layers that of varying sizes that led to a final softmax layer. This led to a model that would overfit and only predict one or two classes. Inspired by [this kaggle notebook](https://www.kaggle.com/code/mohamedchahed/human-emotion-detection), I changed to the convolutional layer's outputs feeding directly into a softmax. I also changed to have pooling layers and added activation functions to the convolutional layers. Changing the number of convolutional layers had little effect on the accuracy after a certain size so the main limitation seems to be model architecture not size.

I used Keras, which is built into Tensorflow, and used their learning rate scheduler to decrease the learning rate when the categorical cross-entropy plateau. I used Adam with an initial learning rate of .02 which decreased by .3 times on plateau.

## 4- Model Evaluation/ Results*
The models would stop improving after about 20 epochs. Below are the accuracy and cross-entropy loss versus epoch graphs for the best-performing model.

![Img](/plots/model6_accuracy.png)
![Img](/plots/model6_cross_entropy.png)

Further training did not improve the cross-entropy loss. Below are the metrics and confusion matrix of the best model.

![Img](/plots/model6_metrics.png)
![Img](/plots/model6_confusion.png)

The overall accuracy was 60% which is similar to other models on Kaggle. The f1-score varies a lot between classes and is pretty correlated to the size of each class. The model has issues predicting the disgust class. The random selection of mispredicted images shows some interesting trends. It seems like some of these mispredictions are mislabeled and some are not human faces. Also, some examples have text over them. This may account for some of the model's inaccuracy.

![Img](/plots/model5_misclassified.png)

## **5- Discussion**