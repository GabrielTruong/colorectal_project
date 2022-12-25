# Colorectal histology - Final Project

Class: Deep Learning
Created: December 3, 2022 5:12 PM
Created By: Gabriel TRUONG
Reviewed: No
Type: Practical Work
Version: 1.0

## 1. Introduction

This work is my final project of the predictive modelling course at EPF. I aim to gather all the knowledge I gain during this course to translate it in this project. The goal here is to implement a full pipeline to solve a supervised learning from scratch, on structured data. 

The dataset I chose is the [colorectal histology](https://www.tensorflow.org/datasets/catalog/colorectal_histology) that contains 5000 images. This dataset represents a collection of textures in histological images of human colorectal cancer. The dataset has been collected from histological samples that have been fully anonymized. These images of formalin-fixed paraffin-embedded human colorectal adenocarcinomas (primary tumors) come from the pathology archive of the following institutes: Institute of Pathology, University Medical Center Mannheim, Heidelberg University, Mannheim, Germany).

This is important to highlight that the initial experiments were made in ethical manner and approved by the institutional ethics board (medical ethics board II, University Medical Center Mannheim, Heidelberg University, Germany; approval 2015-868R-MA). 

The institutional ethics board waived the need for informed consent for this retrospective analysis of anonymized samples. All experiments were carried out in accordance with the approved guidelines and with the Declaration of Helsinki.

Colorectal cancer is one of the widely happening cancers among men and women. This cancer, which is also known as bowel cancer, affects the human large intestine, especially the rectum or colon. Our goal is to to detect colorectal cancer by classifying the histological images among 8 possible labels. By classifying these images, our model will provide an early diagnosis to colorectal cancer and doctors can use it to plan an early and efficient treatment.

All the implementation can be found on my Github. All the results and plots linked to the different models can be found in the annex section.

[GitHub - GabrielTruong/colorectal_project](https://github.com/GabrielTruong/colorectal_project)

## 2. Machine Learning Development

The pipeline of developping the model will remain simple. We will go from a very simple Multi Layer Perceptron Neural Network to iteratively increase the complexity of the model until reaching a good generalization on the training set. Next, we will try to handle the overfitting with different techniques (regularization, data augmentation…). The development process is inspired by the one described in the *****************************Machine Learning Specialization***************************** of DeepLearning.AI.

![Figure 1 - Neural networks and bias variance from Machine Learning Specialization of [DeepLearning.AI](http://DeepLearning.AI) ](assets/DeepLearningAI_bias.png)

Figure 1 - Neural networks and bias variance from Machine Learning Specialization of [DeepLearning.AI](http://DeepLearning.AI) 

![Figure 2 - Iterative loop of ML development from Machine Learning Specialization of [DeepLearning.AI](http://DeepLearning.AI) ](assets/DeepLearningAI_iterative.png)

Figure 2 - Iterative loop of ML development from Machine Learning Specialization of [DeepLearning.AI](http://DeepLearning.AI) 

## 3. **************************************************Exploratory Data Analysis**************************************************

This step will helps us understand the dataset. We will take a look at a sample of images for each label. We will check how balanced is the dataset. Prior to this EDA, we collected the data and split it into a training set and a validation set.

First we retrieved the 8 different labels from the dataset which are:

- Tumor
- Stroma
- Complex
- Lympho

- Debris
- Mucosa
- Adipose
- Empty

Now that we have the different labels, we can see also that the classes are well balanced in *[Figure 3](https://www.notion.so/Colorectal-histology-Final-Project-851287e04e7e4aa7a15f3defd07d93d0)* both in the training and validation set.

![Figure 3 - Both training and validation sets are well balanced](/assets/Balanced_classes.png)

Figure 3 - Both training and validation sets are well balanced

All the images have the same shape that is (150,150,3). Finally let’s take a look at a sample of the images to know how they look like right below.

![Figure 4 - Samples of the dataset labeled](/assets/image_sample.png)

Figure 4 - Samples of the dataset labeled

## 4. **Model development**

In this step, we will iteravely develop model from a simple baseline model to a complex and suitable one.

To evaluate the model we will use a single shuffle split and since the data is well-balanced, we decided to use different metrics. First, we will use the **accuracy** to monitor the training alongside the losses. Since our dataset is well-balanced it will enables us to easily monitor our training. Then we also evaluate the ***confusion matrix*** to see how the models actually predicts. In the ***Practical Guideliness of the Hands-On-ML book,*** the authors give a default set of parameters, we just follow them. Hence all the models use `relu` activation function (instead of `elu`) for the hidden layers and the `adam` optimizer. This will help us to only focus on the architecture at a larger level. 

### 4.1 Baseline model

For this baseline model, we used the architecture from a lab on the Fashion MNIST dataset.  The idea behind this first model is to quickly implement a model that works and can served as the base of our iteration development process. As we can see in *[Table](https://www.notion.so/Colorectal-histology-Final-Project-851287e04e7e4aa7a15f3defd07d93d0) 1,* the accuracy and the F1-score are low both on the training and the validation sets. 

The confusion matrix (***Figure 5.1***) shows that the model only predicts among few labels. The model is having a hard time to predict all labels. We can conclude that this architecture is not complex enough. 

### 4.2 Convolutional Neural Network

Since [Alex Krizhevsky](https://en.wikipedia.org/wiki/Alex_Krizhevsky) won the ImageNet competition in 2012, Convolutional Neural Network is state of the art architecture for image classification. Hence, we will add complexity to our previous model by using a CNN architecture inspired by the AlexNet.

We can see from plots showing the training and validation losses/accuracy ***(Figure 6.2)*** that for few epochs the model had a good dynamic. But quickly the validation accuracy dropped while the training accuracy kept increasing. Hence our goal of getting a more complex model has been reached. The confusion matrix (********Figure 5.2)******** shows that the model is able to classify on all classes but still make mistakes on the validation set. Signs of overfitting are no doubt present and we’ll need to reduce it.  

As described in *[Figure 1](https://www.notion.so/Colorectal-histology-Final-Project-851287e04e7e4aa7a15f3defd07d93d0),* one way to prevent a model from overfitting is by getting more data. Although we only have 5000 images, we can mock more images by using ***data augmentation.*** We will try this method for the next method. We could also try to add ***Dropout*** layers or adding ***L1/L2 regularization terms.*** 

### 4.3 Data Augmentation CNN

This 3rd model has now `RandomFlip` and `RandomRotation` layers that serve for data augmentation. These layers will help us having more different images and also prevent overfitting.

While the training accuracy dropped a little, we now reach 80% accuracy (see ***Table 1)*** on both training and validation sets. The data augmentation layers seemed to have worked perfectly. The confusion matrix (*****Figure 5.3)***** exhibits model improvements. Now the model correctly classifies labels that the previous model made error on.

However we can try to force the convergence a little bit by adding few epochs on this model. We will also add an ************early stopping callback************ to avoid overfitting and unecessary long computation time. If the next modifications doesn’t make any improvement, we can stick to this already highly accurate model. 

### 4.4 More epochs and Early Stopping

With this last model, we tried to train on more epochs to see if we could continue to converge. We set the `period` parameter to 5 for our early-stopping callback. That way, if the model doesn’t improve after 5 epochs then the model stops and we only keep the weights of the best version. 

However, the model stopped at the 28th epoch and the accuracy of the model is below the previous one. The model might have stopped too early. 

## 5. Conclusion

Among these 4 models, the first data augmented CNN would be the one to use. It shows very good accuracy on both sets without signs of overfitting. The early stopped has a lower accuracy and stops approximately to the 30th epoch so there is no improvement on the computational time nor the accuracy. 

Next, we can try to add depth and width to the model but we can expect to again to encounter overfitting. It would be the occasion to add **Dropout** layers or **regularization** terms. Then we would also need to dig further in the hyper-parameters (ie. learning rate, activation functions, batch size etc.)

Since the computational time remains acceptable, we could conduct a **Random Search** on several hyper-parameters to win on accuracy.

## 6. Annex

This sections gathers all the figures and tables referenced in the article for easy comparison between models.

### 6.1 Benchmark Accuracy

| Metrics | Baseline Accuracy | 1st CNN Accuracy | Augmented CNN Accuracy | Early-stopped CNN Accuracy |
| --- | --- | --- | --- | --- |
| Training | 0.255 | 0.982 | 0.847 | 0.8055 |
| Validation | 0.240 | 0.646 | 0.812 | 0.791 |

Table 1- Evaluation metrics for the different models

### 6.2 Confusion matrices

![Figure 5.1 - Confusion Matrix Baseline model](/assets/cm_baseline.png)

Figure 5.1 - Confusion Matrix Baseline model

![Figure 5.2 - Confusion Matrix 1st CNN model](/assets/cm_cnn.png)

Figure 5.2 - Confusion Matrix 1st CNN model

![Figure 5.3 - Confusion Matrix data augmented model](/assets/cm_augmentation.png)

Figure 5.3 - Confusion Matrix data augmented model

![Figure 5.4 - Confusion Matrix early stopped model](/assets/wider_cm.png)

Figure 5.4 - Confusion Matrix early stopped model

### 6.3 Plots loss vs accuracy

![plot_baseline.png](/assets/plot_baseline.png)

Figure 6.1 - Baseline model not complex enough 

![plot_cnn.png](/assets/plot_baseline.png)

Figure 6.2 - Signs of overfitting for first CNN model 

![plot_data_augmentation.png](/assets/plot_data_augmentation.png)

Figure 6.3 - Fits just right

![plot_wider.png](/assets/plot_wider.png)

Figure 6.4 - Stopped too early