# Predicting Fashion MNIST Classes

This project showcases a collection of TensorFlow models tailored for the MNIST Fashion dataset, a benchmark in image classification. The goal is to understand how various architectures affect a model's predictive power and generalisability. We will be performing an in-depth performance analysis of each model based on various metrics such as accuracy, precision, and F1 score to select the most robust approach.

## Installation

This script requires Python 3 and the following dependencies:

- Keras (import Fashion MNIST dataset)
- Matplotlib (plotting results)
- NumPy (manipulating arrays and apply mathematical operations)
- Scikit-learn (model evaluation and performance metrics)
- Seaborn (statistical data visualisation)
- TensorFlow (build and train deep learning models)

```bash
pip install keras
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install seaborn
pip install tensorflow
```

## Usage

To view the project, follow these steps:
- Clone the repository or download it as a zip folder and extract all files.
- Ensure you have installed the required dependencies.
- Run the Fashion_MNIST_Models.ipynb notebook.

## Methodology

**Data Preprocessing**
   - Read in the MNIST Fashion dataset (via Tensorflow), which contains 70,000 images of 10 different clothing classes such as Coats, Trousers, Dresses etc.
   - Split the dataset into training, validation, and test sets.
   - Normalise and reshape image datasets to fit model input requirements.

**Model Selection**

Several model architectures were considered:
   - Model 1: Simple Neural Network (Baseline), with 2 fully-connected hidden layers.
   - Model 2: Deep Neural Network (DNN) - No Regularisation, with 3 fully-connected hidden layers.
   - Model 3: Deep Neural Network (DNN) - Regularisation, with 3 fully-connected hidden layers and L2-Regularisation.
   - Model 4: Convolutional Neural Network (CNN), with 3 convolutional layers and 2 dropout layers.
   - Model 5: Recurrent Neural Network (RNN), with 2 LSTM layers and L2-Regularisation.
   - Model 6: Residual Network (ResNet), with 1 convolutional layer, 2 residual blocks, and a fully-connected hidden layer with L2-Regularisation.

**Evaluation Metrics**
   - The primary metric used to evaluate model performance was accuracy on the test set.
   - F1 score, precision, and recall were also computed to assess the model's ability to correctly classify each fashion category.

## Results
All models performed admirably, achieving accuracies surpassing 83% on the test set. Notably, **ResNet (Model 6)** emerged as the standout performer with an impressive accuracy of **91.59%**, closely followed by **CNN (Model 4)** at **91.27%**.

CNN showcased its effectiveness in image classification tasks by leveraging localized feature extraction through convolutional layers, which proved instrumental in discerning intricate patterns within the MNIST Fashion dataset. Meanwhile, ResNet demonstrated robust capabilities in mitigating overfitting and exhibited strong performance across all classes, particularly excelling in differentiating between upper body garments such as shirts, pullovers, coats, and t-shirts/tops.

Future work to enhance the ResNet model include fine-tuning critical hyperparameters such as the L2-Regularisation penalty, initial learning rate, and batch size, optimising configurations of convolutional layers, exploring adjustments in network depth, and refining the number of residual blocks and skip connections. These refinements aim to consolidate ResNet's strengths and extend its applicability across diverse domains, pushing the boundaries of its capabilities in real-world applications.