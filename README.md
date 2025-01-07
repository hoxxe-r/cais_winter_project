# CAIS Winter Project
Aidan Parris ajparris@usc.edu

## 1 - Project Presentation
I chose to do the facial expression recognition task. Because this dataset is images, I used a convolutional neural network that uses convolutional layers with 3x3 filters and batch normalization. There are three sections, each with increasing numbers of filters and with max-pooling layers after.

## 2 - Dataset
This dataset contains seven expression classes: anger, disgust, fear, happy, sad, surprise, and neutral. There are 28,709 images in the train category and 7,178 images in the test category giving a train/test split of 0.8. The smallest category is disgust with 547 photos and the largest category is happy with 8,989 images. Other than these two classes, the dataset is fairly balanced. The lack of examples of disgust led to issues in achieving high accuracy for this class. Each image is 48 by 48 pixels and grayscale in the range \[0, 255].
![Img](/plots/class_count.png)

## 3 - Model Development and Training
Initially, the model I had had a couple of convolutional layers that had ~1000 outputs when flattened then those fed into a neural network with dense layers of varying sizes that led to a final softmax layer. This led to a model that would overfit and only predict one or two classes. Inspired by [this Kaggle notebook](https://www.kaggle.com/code/mohamedchahed/human-emotion-detection), I changed to the convolutional layer's outputs feeding directly into a softmax. I also changed to have pooling layers and added activation functions to the convolutional layers. Changing the number of convolutional layers has little effect on the accuracy after a certain size so the main limitation seems to be model architecture not size.

After this model, I attempted to improve the accuracy by having the outputs of the CNN feed into a neural network. At first, this led to similar overfitting, however, after decreasing the learning rate, these models began to outperform the CNN into softmax. Increasing the filters from 64 to 100 fine-tuning the size of the NN, and adding a dropout layer after the CNN led to the best-performing model. [This](/plots/hybrid_low_lr2_model.png) is the model's architecture.

I used Keras, which is built into TensorFlow. This offered easy model creation, training, and analysis. I used the built-in learning rate scheduler to decrease the learning rate on plateau and used categorical cross entropy as the model's loss metric. The final model used Adam with a learning rate of $5*10^{-5}$ and was initially trained for 15 epochs. A further 7 epochs did not improve the results further.

## 4 - Model Evaluation/ Results
The models would stop improving after about 20 epochs. Below are the accuracy and cross-entropy loss versus epoch graphs for the best-performing model.

![Img](/plots/hybrid_low_lr2_accuracy.png)
![Img](/plots/hybrid_low_lr2_cross_entropy.png)

Further training did not improve the cross-entropy loss, as evident by a second training in which the [accuracy](/plots/hybrid_low_lr2_2_accuracy.png) and [cross entropy](/plots/hybrid_low_lr2_2_cross_entropy.png) begin to diverge between the training and validation datasets. This indicates overfitting to the training dataset. Below are the metrics and confusion matrix of the best model.

![Img](/plots/hybrid_low_lr2_metrics.png)
![Img](/plots/hybrid_low_lr2_confusion.png)

The accuracy was 63% which is similar to other models on Kaggle. The f1-score varies a lot between classes. As shown by the disgusting class having the lowest support and f1-score, the model's ability to predict classes is correlated to the classes' size. The random selection of mispredicted images shows some interesting trends. It seems like some of these mispredictions are mislabeled and some are not human faces. Also, some examples include text which may negatively impact the model. 

![Img](/plots/hybrid_low_lr2_misclassified.png)

## 5 - Discussion
Overall, I was fairly successful at implementing a CNN that was capable of detecting facial expressions. A comparable accuracy to other models on Kaggle was achieved. Some areas for further research are trying new model architectures and potential skip layers to prevent the model from losing broad information about faces. Initially, I had issues adding a neural network to the end of the CNN as the models would overfit and [only predict one class](/plots/small_cnn2_hybrid_lowlr3_confusion.png). I learned that adding an NN required lowering the learning rate by about two orders of magnitude to prevent this overfitting. Also, adding more filters to the CNN increased the model's accuracy. However, there were diminishing returns and required more computing time for training and computing. Further increasing filters is a potential avenue to further increase the model's accuracy.