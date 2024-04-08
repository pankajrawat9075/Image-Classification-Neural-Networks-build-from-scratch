# Goal 1 - train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters
    Assignment 2 part A Course Assignment of CS6910 Deep Learning IIT Madras
## Abstract<br/>
This assignment involves training a **Convolutional Neural Network (CNN) model** from scratch using the **iNaturalist12k dataset**. The goal is to learn how to **tune hyperparameters** for optimal model performance and **visualize filters** to **gain insights** into the model's learning process. 
The process involves 
- setting up the environment,
- preparing the dataset,
- defining the model architecture,
- training and evaluating the model,
- tuning the hyperparameters, and
- visualizing the filters.
- 
## Dataset<br/>
The **inaturalist12k** dataset is a large-scale image classification dataset that contains over **12,000 species of plants and animals**, with **each species** having **at least 1,000 images**. The dataset consists of high-resolution images, with each image being labeled with the corresponding species name. The dataset is divided into training, validation, and testing sets, with the **training set containing 80% data**, the **validation set containing 20% data**, and the **testing set containing 1,930 species and 158,370 images**.

## Objective<br/>
dataset and achieve high accuracy by tuning hyperparameters. Additionally, the assignment aims to visualize the learned filters to gain insights into how the model processes the images. Through this assignment, we will learn how to fine-tune hyperparameters, experiment with different model architectures, and interpret the results to improve model performance.

## Folder Structure<br/>
these below files contain the required code for second assignment part B<br/>
DL Assignment_2_ques_1(CNN).ipynb<br/>
question-4-part-a.ipynb<br/>
train.py<br/>
README.MD<br/>

## Results<br/>
The best training accuracy on inaturalist12k dataset achieved is *43%* while validation accuracy is *37%*. The explanation and results of subproblems 
can be accessed ([https://api.wandb.ai/links/cs22m010/3z0c1j6z](https://api.wandb.ai/links/iitmadras/3pav9adf))

# Goal 2 - finetune a pre-trained model just as you would do in many real-world applications
Assignment 2 part B Course Assignment of CS6910 Deep Learning IIT Madras
## Abstract<br/>
This assignment involves fine-tuning the pre-trained GoogleNet model on the inaturalist12k dataset. The inaturalist12k dataset is a large-scale image classification dataset that contains over 12,000 species of plants and animals, with each species having at least 1,000 images. The goal of this project is to fine-tune the GoogleNet model to accurately classify images from the inaturalist12k dataset.
## Dataset<br/>
The inaturalist12k dataset is a large-scale image classification dataset that contains over 12,000 species of plants and animals, with each species having at least 1,000 images. The dataset consists of high-resolution images, with each image being labeled with the corresponding species name. The dataset is divided into training, validation, and testing sets, with the training set containing 80% data, the validation set containing 20% data, and the testing set containing 1,930 species and 158,370 images.
## Objective<br/>
The objective of this project is to fine-tune the pre-trained GoogleNet model to accurately classify images from the inaturalist12k dataset. Fine-tuning involves taking a pre-trained model and adapting it to a new dataset by re-training the last few layers of the model. By fine-tuning the GoogleNet model on the inaturalist12k dataset, we aim to achieve high accuracy in classifying images from the dataset. This can have practical applications in various fields, including ecology, conservation biology, and agriculture.
## Folder Structure<br/>
these below files contain the required code for second assignment part B<br/>
googlenet-1.ipynb.ipynb<br/>
train.py<br/>
README.MD<br/>
## Results<br/>
The best training accuracy on inaturalist12k dataset achieved is *73.250000%* while validation accuracy is *69.934967%*. The explanation and results of subproblems 
can be accessed (https://api.wandb.ai/links/cs22m010/3z0c1j6z)
