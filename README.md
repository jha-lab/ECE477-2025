# ECE477-2025

## Welcome to ECE 477 Smart Healthcare class!

This class covers the following topics:
Introduction to smart healthcare; health decision support system; wearable medical sensors and deep neural network based disease detection; continual learning based multi-headed neural networks for multi-disease detection; interpretability through differentiable logic networks; interpretability through conformal predictions; medical images and convolutional neural network based disease detection; natural language processing for healthcare; foundation models for healthcare; counterfactual reasoning based personalized medical decision-making.

All assignments are built on papers you will read for the class.
Each of these coding tasks exemplifies the paper discussed this week, and reproduces a simplified version of the smart healthcare framework in this paper.
The assignments introduce you to the latest advances in Machine Learning (broadly) and especially Deep Learning.

Useful reference to ML concepts: [ML Glossary from Google](https://developers.google.com/machine-learning/glossary#multi-class)

## Course staff

* Instructor: [Niraj K. Jha](https://www.princeton.edu/~jha/)
* TAs: Margarita Belova, Jiaxin Xiao

## Timings

* Lectures: M/W 11:00-12:20pm 
* Office hours:
    * Margarita Belova: Tu: 3-4pm, Th: 10-11am (EQuad B321)
    * Jiaxin Xiao: Mo: 10-11am, Wed 2-3pm (EQuad B321)

## Assignment descriptions

Each assignment is worth 20 points.

**The data for assignments can be accessed on [shared disk at Google Drive](https://drive.google.com/drive/u/0/folders/0ABIZHKB-QPnRUk9PVA)**


**Assignment 1: ML classifiers**

For this project, you will train and compare various classifiers (decision tree, k-nearest neighbor, Naive Bayes, and logistic regression) to determine whether a patient has breast cancer.
We will use the Diagnostic Wisconsin Breast Cancer dataset from the UCI machine learning repository (see details [here](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)).
The purpose of this first assignment is to recall the basic ML classifiers, Python data science libraries (Numpy, Pandas, Sklearn), and ML concepts (training-validation-test splits, training accuracy, validation accuracy), that we will draw upon in further assignments.

**Assignment 2: SCANN**

For this project, you will train compact neural networks with a mutating architecture that adapts to data during the training process. You will work with an arrhythmia dataset.

Your goal is to implement SCANN Scheme A. This is a constructive approach. We start with a seed architecture that has a small number of hidden neurons. Using an iterative process, we apply connection growth and neuron growth to grow the network size. We have a maximum number of neurons that we cannot exceed (this is one of the hyperparameters).

**Assignment 3: TUTOR**

For this project, you will implement the TUTOR framework to train accurate deep neural networks with limited data and optimized computational resources. The workflow includes generating synthetic data using KDE and GMM methods, validating it with semantic integrity classifiers, labeling synthetic data with a random forest classifier, and training baseline models on real data. You will then apply Scheme A: pretraining on synthetic data followed by final training on real data, to demonstrate the benefits of this approach.

**Assignment 4: CovidDeep**

For this project, you will train deep neural networks on data obtained from COVID-19 patients to predict if the patient is healthy or has COVID-19. This includes three steps:
(a) Generate synthetic data using the TUTOR methodology.
(b) Use synthetic data to pre-train a neural network, then finish training on real data.
(c) Apply the SCANN "grow-and-prune" neural network synthesis paradigm to compress the original network.

After completing grow-and-prune synthesis, you should obtain a compact neural network that has better accuracy than before using this step.

**Assignment 5: SoDA**

For this project, you will implement a framework for stress detection based on physiological signals. The experimental data are collected from 32 individuals using wearable medical sensors. The dataset features are ECG, Galvanic Skin Response (GSR), Respiration, Blood Oximeter, and Blood Pressure. To reduce the data size, you will employ Principal Component Analysis, which is a versatile unsupervised dimensionality reduction technique. With the reduced dataset, you will train and evaluate k-NN and SVM models with radial basis functions for binary stress classification.

**Assignment 6: DOCTOR**

DOCTOR is a framework that enables continual learning for disease detection based on wearable medical sensor data.

In this exercise, you will recreate some of the experiments described in the DOCTOR paper. You will perform domain-, class-, and task-incremental learning using a multilayer perceptron (MLP) model that detects diabetes and mental health disorders using replay-based continual learning methods.


**Assignment 7: DLN**

For this project, you will train and evaluate a DLN for classification using the Heart Disease Kaggle dataset. It includes the steps of data preparation, including preprocessing, scaling, and feature reordering. It then includes how to train a DLN model, evaluate its performance, and visualize the learned network.
See the Jupyter notebook for more details.

**Assignment 8: CONFINE**

In this assignment, you will implement the CONFINE algorithm, a versatile framework that generates prediction sets with statistically robust uncertainty estimates. The assignment will guide you through the following steps.

(a) Train a DNN on the CovidDeep Dataset. The model should classify patients into three health
categories: healthy, asymptomatic, and symptomatic. The DNN will be used to extract feature embeddings that will be used by CONFINE.
(b) Compute Nonconformity Scores: Implement the CONFINE nonconformity score using cosine
distance.
(c) Calculate p-Values: For each test sample, compute p-values based on the nonconformity
scores. You will calculate p-values for each class using both the CONFINE and CONFINE-classwise methods, which differ in how calibration samples are used.
(d) Make Predictions and Compute Confidence: For each test sample, you will use the p-values
to create a prediction set, determine the final prediction, and calculate the model’s credibility
and confidence.
(e) Evaluate the Model: Finally, you will evaluate the model’s performance using accuracy,
correct efficiency, and coverage.

**Assignment 9: LSTM**

In this coding assignment, you will develop an LSTM-based classifier for diabetes diagnosis (healthy vs. unhealthy) using the DiabDeep dataset. The DiabDeep dataset contains physiological signals obtained with wearable sensors from 52 participants.

Through the assignment, you will implement key components of a neural model based on the DiabDeep paper. You will be tasked with adding gates to a hidden-layer LSTM (H-LSTM) architecture, in which each control gate incorporates an additional hidden layer with a ReLU activation to improve learning capacity. Then, you will complete the H-LSTM Classifier model—responsible for predicting healthy versus unhealthy diagnoses—by adding the missing layers and forward pass calculations, and finally, train the model on the DiabDeep dataset.

**Assignment 10: LLM**

This assignment provides hands-on experience querying different Large Language Models (LLMs) through a cloud API provider (Nebius AI). You will focus on a simplified medical Question Answering task, using sample questions from MedMCQA, a large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions.

The goal of this assignment is to develop practical skills in interacting with LLM APIs, comparing the capabilities of different open-source models (including those potentially fine-tuned for biomedical domains), exploring the impact of generation parameters, and evaluating their potential and limitations in the healthcare context.


## License

BSD-3-Clause. 
Copyright (c) 2025, JHA-Lab.
All rights reserved.

See License file for more details.
