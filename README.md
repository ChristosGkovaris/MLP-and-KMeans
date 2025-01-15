# MLP-and-KMeans
Welcome to the "MLP-and-KMeans" repository. This repository includes implementations of K-Means Clustering, visualization tools, and a neural network project for classification tasks. The projects demonstrate key concepts in data clustering, visualization, and machine learning using Java.


## Overview
This repository provides implementation of K-Means Clustering with a focus on clustering error and convergence. Visualization of
clustering results using a GUI. A neural network for classification tasks, with configurable architectures and visualization capabilities.


### K-Means Clustering
- Description: Implementation of the K-Means algorithm to cluster data points in a 2D space with different cluster sizes.
- Key Features: Supports multiple cluster counts (`M` values: 4, 6, 8, 10, 12). Random centroid initialization with
  convergence detection. Outputs clustering results to a CSV file, including clustering error and coordinates. Handles 1000
  data points distributed across predefined regions and random noise.


### K-Means Visualization
- Description: GUI-based visualization of the clustering results from the CSV file generated by the K-Means Clustering project.
- Key Features: Displays clusters with unique colors and centroids marked by stars (`*`). Provides an interactive
  plot with labeled axes for better analysis. Scales plot dimensions automatically based on the data.


### Neural Network Project
- Description: A multi-layer perceptron (MLP) neural network designed for classification tasks with
  configurable hidden layers and activation functions.
- Key Features: Generates synthetic data for training and testing. Supports multi-class classification
  using softmax output activation. Configurable layer sizes (`H1`, `H2`, `H3`) and activation functions
  (`ReLU`, `tanh`). Uses cross-entropy loss and backpropagation for training. Outputs results (accuracy)
  for various configurations to a CSV file.


### Neural Network with Visualization
- Description: Visualization of MLP configurations and their respective accuracies.
- Key Features: Plots bar graphs of accuracy against different MLP configurations. Provides interactive
  GUI with labeled configurations. Highlights activation functions with color-coded bars (`ReLU` in red, `tanh` in blue).


## How to Run
- Clone the repository:
  ```bash
     git clone https://github.com/ChristosGkovaris/MLP-and-KMeans.git
     cd MLP-and-KMeans


## Collaboration
This project was a collaborative effort. Special thanks to [SpanouMaria](https://github.com/SpanouMaria), for their significant contributions to the development and improvement of the game.
