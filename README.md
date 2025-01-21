# ECE477-2025

## Welcome to ECE 477 Smart Healthcare class!

This class covers the following topics:
Introduction to smart healthcare; health decision support system; wearable medical sensors and deep neural network based disease detection; continual learning based multi-headed neural networks for multi-disease detection; interpretability through differentiable logic networks; interpretability through conformal predictions; medical images and convolutional neural network based disease detection; natural language processing for healthcare; foundation models for healthcare; counterfactual reasoning based personalized medical decision-making.

All assignments are built on papers you will read for the class.
Each of these coding tasks exemplifies the paper discussed this week, and reproduces a simplified version of the smart healthcare framework in this paper.
The assignments introduce you to the latest advances in Machine Learning (broadly) and especially Deep Learning.

Useful reference ML concepts: [ML Glossary from Google](https://developers.google.com/machine-learning/glossary#multi-class)

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

**Assignment 4: CovidDeep**

For this project, you will train deep neural networks on data obtained from COVID-19 patients to predict if the patient is healthy or has COVID-19. This includes three steps:
(a) Generate synthetic data using the TUTOR methodology.
(b) Use synthetic data to pre-train a neural network, then finish training on real data.
(c) Apply the SCANN "grow-and-prune" neural network synthesis paradigm to compress the original network.

After completing grow-and-prune synthesis, you should obtain a compact neural network that has better accuracy than before using this step.



## License

BSD-3-Clause. 
Copyright (c) 2023, JHA-Lab.
All rights reserved.

See License file for more details.
