# XAI-deeplift-vs-ig-project

LSTM model based closing price predition for Bitcoin dataset with explanations using DeepLIFT and Integrated Gradients.

# Description

My aim here is to compare explanations using DeepLIFT and Integrated Gradients to see whether they converge on the same temporal features and from this deduce their reliability.
The code consists of a model class where i implement a custom LSTM model using pytorch and following a tutorial (https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/)

# Dataset

The csv is not included because of it's big size here is the download link:
https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
Just place it in the same directory where the rest of the files sit.

# Dependencies

In order to run this you need the following libraries:
pip install pandas scikit-learn numpy torch matplotlib captum

# How to run:

python main.py
