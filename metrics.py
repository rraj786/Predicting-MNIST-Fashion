'''
    The following script contains functions to compute various metrics using model outputs
    and evaluate performance.
    Author: Rohit Rajagopal
'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def classification_report_ad(true_labels, predicted_labels):

    """
        Generate a classification report of key metrics for model evaluation.
    
        Args:
            - true_labels (array-like): A 1D array of true class labels.
            - predicted_labels (array-like): A 1D array of predicted class labels.
    
        Returns:
            None
    """
    
    # Generate classification report for micro and macro-averages
    report = classification_report(true_labels, predicted_labels, digits = 4)

    # Display report
    print(report)

    return

def confidence_distribution(output_probabilities):

    """
        Plot the distribution of confidence scores for predicted labels.
    
        Args:
            - output_probabilities (array-like): A 1D array containing confidence scores for predicted labels.
    
        Returns:
            None
    """

    # Plot and display histogram
    plt.hist(output_probabilities, bins = 10, color = 'skyblue', edgecolor = 'black')
    plt.xlabel('Confidence Scores')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Scores for Predicted Labels')
    plt.show()

    return

def confusion_matrix_ad(true_labels, predicted_labels, num_classes, classes):
    
    """
        Generate and plot a confusion matrix.
    
        Args:
            - true_labels (array-like): A 1D array of true class labels.
            - predicted_labels (array-like): A 1D array of predicted class labels.
            - num_classes (int): Number of classes in the classification task.
            - classes (list): List of actual class names for the classification task.
    
        Returns:
            None
    """

    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot and save confusion matrix
    plt.figure(figsize = (10, 8))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', cbar = True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(np.arange(num_classes) + 0.5, labels = classes, fontsize = 8)
    plt.yticks(np.arange(num_classes) + 0.5, labels = classes, fontsize = 8)
    plt.title('Confusion Matrix')
    plt.show()

    return

def training_progress(epochs, train_accuracies, train_losses, val_accuracies, val_losses):

    """
        Plot training progress over epochs.
    
        Args:
            - epochs (list): List of epoch numbers.
            - train_accuracies (list): List of accuracy values corresponding to each epoch for the training set.
            - train_losses (list): List of loss values corresponding to each epoch  for the training set.
            - val_accuracies (list): List of accuracy values corresponding to each epoch for the validation set.
            - val_losses (list): List of loss values corresponding to each epoch  for the validation set.
    
        Returns:
            None
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    # Plot accuracies
    ax1.set_title('Progress of Accuracies')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.plot(epochs, train_accuracies, color = 'tab:blue', label = 'Training Set')
    ax1.plot(epochs, val_accuracies, color = 'tab:red', label = 'Validation Set')
    ax1.legend(loc = 'lower right')

    # Plot losses
    ax2.set_title('Progress of Losses')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.plot(epochs, train_losses, color = 'tab:blue', label = 'Training Set')
    ax2.plot(epochs, val_losses, color = 'tab:red', label = 'Validation Set')
    ax2.legend(loc = 'upper right')

    fig.tight_layout()
    plt.suptitle('Accuracy and Loss per Epoch for Training and Validation Sets', y = 1.05)
    plt.show()

    return
