# Image Classification with Transfer Learning

# Overview

-> This project demonstrates image classification using transfer learning with MobileNetV2 on the CIFAR-10 dataset. The model leverages a pre-trained convolutional neural network (CNN) and fine-tunes the last 20 layers to improve performance. Data augmentation, dropout, and batch normalization techniques are implemented to enhance generalization.

# Approach

1. Data Preprocessing

-> Loaded the CIFAR-10 dataset.

-> Normalized pixel values between 0 and 1.

-> Applied extensive data augmentation (rotation, shifting, shear, zoom, and flips).

-> Split data into training, validation, and test sets.

2.  Transfer Learning

-> Used MobileNetV2 as the pre-trained model (excluding top layers).

-> Added custom dense layers with batch normalization and dropout.

-> Fine-tuned the last 20 layers while keeping the rest frozen.

3. Training & Evaluation

-> Trained the model using Adam optimizer with a learning rate of 0.0001.

-> Evaluated performance using accuracy and loss metrics.

-> Generated a confusion matrix to analyze misclassifications.

-> Plotted loss and accuracy curves to track training progress.

# How to Run

1. Install dependencies:
  ->  pip install tensorflow numpy matplotlib seaborn scikit-learn

   
2. Run the script:
          ->    python image_classification.py
3.  The model will train, evaluate, and generate performance visualizations

# Challenges Encountered

-> Fine-tuning required careful selection of trainable layers to avoid overfitting.
->  Data augmentation significantly improved model generalization but increased training time.
->  Hyperparameter tuning (dropout rates, learning rate) played a crucial role in model stability.

# Results
-> Achieved competitive accuracy on the CIFAR-10 test set.
-> Model successfully classifies images into 10 categories with improved robustness.
-> Visualization of training history and confusion matrix provides insights into performance.

# Future Improvements
-> Experiment with different pre-trained models like ResNet50 or EfficientNet.
-> Implement learning rate scheduling for better convergence.
->  Apply advanced regularization techniques to reduce overfitting further.


