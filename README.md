<h1 align="center">
AI-Projects
</h1>

![image](https://github.com/AryanBastani/asd/assets/101284002/6925412d-7bac-461c-a375-22494cabb28c)


## Introduction

Welcome to the AI course projects repository. This collection showcases a variety of key concepts and techniques in artificial intelligence. Each project delves into a specific area of AI, providing hands-on experience and practical applications. These projects are designed to help you understand and implement both foundational and advanced AI algorithms.

Below are brief summaries of each project included in this repository:

---

### Project 1: Genetic Algorithms
![image](https://github.com/AryanBastani/asd/assets/101284002/2293499d-9ed1-46aa-9bdd-b0ef17c16000)


**Introduction:**
Genetic algorithms are inspired by natural selection and use ideas such as mating, mutation, and selection to model and solve optimization problems by iteratively improving solutions based on their fitness.

**Problem Description:**
Solve the Knapsack Problem, deciding which items (snacks) to take on a picnic under constraints of weight, value, and diversity. The goal is to maximize the total value while adhering to these constraints.

**Implementation Steps:**

1. **Define Genes and Chromosomes:** Chromosomes represent potential solutions, composed of genes that encode decisions about which snacks to take and in what quantities.
2. **Generate Initial Population:** Create an initial set of random chromosomes (solutions).
3. **Fitness Function:** Define a fitness function to evaluate how well each chromosome satisfies the problem constraints and objectives.
4. **Crossover and Mutation:** Implement crossover (combining pairs of chromosomes) and mutation (randomly altering genes) to create new chromosomes for the next generation.
5. **Genetic Algorithm Execution:** Run the genetic algorithm through multiple generations, selecting the best chromosomes to form new populations, aiming to improve the solutions iteratively.
6. **Evaluate Results:** Test the algorithm with different inputs and refine the parameters to ensure optimal performance.

**Some of Questions:**

1. Impact of small or large initial populations.
2. Effect of increasing population size per generation on accuracy and speed.
3. Comparison of crossover and mutation operations.
4. Strategies to expedite reaching a solution.
5. Issues when chromosomes stop evolving and how to address them.
6. Solutions for situations where the problem has no feasible answer.

---

### Project 2: Hidden Markov Models (HMM)
![image](https://github.com/AryanBastani/asd/assets/101284002/b75566e0-6907-4f68-99ea-7baa7824a233)


**Introduction:**
Hidden Markov Models (HMM) are powerful tools for modeling time-series data and pattern recognition, particularly in dynamic systems with uncertainty. They are widely used in speech recognition.

**Problem Description:**
Develop a speech recognition system for digits using HMM, utilizing a provided dataset of spoken digits from six speakers.

**Implementation Steps:**

1. **Data Preprocessing and Feature Extraction:** Preprocess audio data to enhance quality and segment it. Extract features such as MFCCs from audio samples.
2. **Understanding HMM:** Define states, observations, and transition/emission probabilities to model the system's behavior.
3. **HMM Implementation:**
   - **With Libraries:** Use libraries like `hmmlearn` to build and train the HMM on the dataset.
   - **From Scratch:** Implement the HMM algorithm manually, including methods for state likelihood, the Expectation-Maximization (EM) step, training, and scoring.
4. **Evaluation and Analysis:** Use metrics like F1 score, recall, precision, and accuracy to evaluate the model's performance. Generate confusion matrices to analyze performance.

**Some of Questions:**

1. Utility of segmentation for this dataset.
2. Detailed study of feature extraction techniques and their interrelationships.
3. Robustness and sensitivity of MFCC features.
4. Advantages and limitations of using MFCCs.
5. Reasons for frame overlap in MFCC calculation.
6. Reasons for using only the first 12-13 MFCC coefficients.
...

---

### Project 3: Clustering
![image](https://github.com/AryanBastani/asd/assets/101284002/b190d0d2-79c4-4af6-95fb-fdc8bdc0b4ae)



**Introduction:**
Clustering involves grouping similar objects based on inherent similarities to discover natural groupings within the data. This technique is useful for applications like customer segmentation, image categorization, anomaly detection, and recommendation systems.

**Problem Description:**
Cluster images of different flower species using clustering algorithms to group them accurately based on their features.

**Implementation Steps:**

1. **Data Preprocessing and Feature Extraction:** Use the pre-trained VGG16 Convolutional Neural Network to extract features from flower images.
2. **Clustering Methods:**
   - **K-Means:** Choose an appropriate K value based on the number of flower categories and cluster the feature vectors.
   - **DBSCAN:** Cluster the feature vectors using density-based clustering.
3. **Dimensionality Reduction:** Use PCA to reduce the dimensionality of the feature vectors for visualization and comparison.
4. **Evaluation and Analysis:** Use homogeneity and silhouette scores to evaluate the clustering results. Compare the performance of K-Means and DBSCAN.

**Some of Questions:**

1. Reasons for feature extraction over raw pixel reading.
2. Summary of three feature extraction techniques from images.
3. Preprocessing steps for preparing images for the model.
4. Comparison of K-Means and DBSCAN, including their pros and cons.
5. Method used to determine the optimal K value in K-Means.
6. Comparison of clustering results from K-Means and DBSCAN.
7. Explanation and function of PCA for dimensionality reduction.
8. Calculation and significance of silhouette and homogeneity scores.
9. Reporting and analysis of clustering performance using these metrics.
10. Suggestions for improving model performance.
...

---

### Project 4: Machine Learning
![image](https://github.com/AryanBastani/asd/assets/101284002/a34a6e07-9030-4920-a057-cbc5712c9b51)


**Introduction:**
Machine Learning models are employed to make predictions based on data. This project focuses on predicting house prices in Boston using various machine learning techniques.

**Problem Description:**
Predict the prices of houses in Boston based on features such as crime rate, number of rooms, and distance to employment centers.

**Implementation Steps:**

1. **Data Familiarization:** Understand the dataset, including the types of features and their significance. Perform basic statistical analysis to identify distributions and outliers.
2. **Data Preprocessing:**
   - **Handling Missing Values:** Implement techniques like mean imputation, median imputation, or removal of missing data.
   - **Feature Scaling:** Apply normalization or standardization to numerical features.
   - **Categorical Features:** Encode categorical features using methods like one-hot encoding or label encoding.
3. **Model Training:**
   - **Linear Regression:** Train a linear regression model and evaluate using metrics like Mean Squared Error (MSE) and R² score.
   - **Decision Trees and Random Forests:** Train decision tree and random forest models. Compare their performance against linear regression.
4. **Model Evaluation and Tuning:** Use techniques like grid search or random search for hyperparameter tuning. Implement k-fold cross-validation to ensure robustness.
5. **Advanced Techniques (Optional):** Explore ensemble methods and feature engineering to enhance model performance.

**Some of Questions:**

1. Methods to handle missing values and their impact.
2. Importance and methods of feature scaling.
3. Differences between categorical and numerical features and their preprocessing techniques.
4. Comparison of linear regression, decision trees, and random forests.
5. Explanation and implementation of hyperparameter tuning.
6. Evaluation metrics used for regression models and their significance.
7. Strategies for model validation and ensuring robustness.
8. Benefits and challenges of ensemble methods.
9. Steps and importance of feature engineering.
10. Reasons for using cross-validation and its effect on model performance.
...

---

### Project 5: Convolutional Neural Networks (CNNs)
![image](https://github.com/AryanBastani/asd/assets/101284002/f0e01101-e105-4359-9b9d-64ef4a8db912)





**Introduction:**
Convolutional Neural Networks (CNNs) are specialized deep learning models designed for processing structured grid data, such as images. They leverage spatial hierarchies in the data to perform tasks like classification, detection, and segmentation.

**Problem Description:**
CNNs address the challenge of recognizing and interpreting patterns in visual data, including tasks such as identifying objects in an image, distinguishing between different scenes, and segmenting parts of an image for further analysis.

**Implementation Steps:**

1. **Convolutional Layers:** Apply convolution operations to detect features like edges, textures, and patterns.
2. **Pooling Layers:** Reduce the spatial dimensions through operations like max pooling or average pooling.
3. **Fully Connected Layers:** Use dense layers to perform high-level reasoning and classification.
4. **Activation Functions:** Introduce non-linearities using activation functions like ReLU.
5. **Training Process:** Train the CNN using a large dataset, optimizing parameters through backpropagation and gradient descent.
6. **Evaluation and Fine-Tuning:** Evaluate the model on validation data, fine-tune hyperparameters, and iterate to improve performance. Use techniques like dropout and data augmentation.

**Some of Questions:**

1. Impact of different types of convolutional layers (e.g., standard, depthwise separable) on model performance.
2. Effects of varying pooling layer parameters on the model's ability to generalize.
3. Influence of architecture depth (number of layers) on the CNN's accuracy and training time.
4. Best practices for selecting and applying activation functions in CNNs.
5. Impact of different optimization techniques (e.g., Adam, SGD) on training and final model performance.
6. Strategies to mitigate overfitting in CNNs, especially with smaller datasets.
...

---

### Project 6: Reinforcement Learning
![image](https://github.com/AryanBastani/asd/assets/101284002/cea9c87a-c12b-4f79-aada-4cc8533a30f3)


**Introduction:**
Reinforcement learning involves an agent exploring and interacting with its environment to gather knowledge and maximize total rewards. This project focuses on designing an AI opponent for the classic Snake game.

**Problem Description:**
**Task 1: Snake (Nostalgia)**
Design an AI opponent for a two-player version of the Snake game. The goal is to grow by eating apples and defeat the opponent. The game ends when one snake's head collides with another snake's body or itself, or when both heads collide, with the longer snake winning.

**Implementation Steps:**

1. **Game Rules Adaptation:** Adapt the classic Snake game rules for two-player mode.
2. **Q-Learning Agent Training:** Train your agent using the Q-learning method to maximize rewards over time.
3. **Observation Space Reduction:** Reduce the large observation space by defining features that describe the environment, such as coordinates of the apple and the opponent snake's head.
4. **Define State and Action Space:** Define the state space based on the reduced observation space and the action space as possible moves (up, down, left, right).
5. **Exploration and Exploitation:** Implement strategies such as Decay Epsilon to balance exploration and exploitation during training.
6. **Hyperparameter Tuning and Model Saving:** Experiment with different iterations and hyperparameters. Save the trained models and plot the reward earned by the model for each episode.

**Competition: Snake (Scoring)**
Evaluate the performance of the trained snake models by competing against each other in a knockout league, with the winner determined based on the best of 101 games.

**Some of Questions:**

1. Impact of reducing the observation space on the agent's performance.
2. Effects of different distance metrics (Euclidean, Manhattan) on the Q-learning algorithm.
3. Influence of the Decay Epsilon strategy on exploration-exploitation balance.
4. Best practices for defining state and action spaces in a large observation environment.
5. Impact of different hyperparameters on training time and performance.
6. Strategies to prevent the agent from overfitting during training.
...

---
