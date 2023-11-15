# Lip Reading - Predicting Sentences from Input Video

## Phase 1 Overview

Welcome to Phase 1 of the Lip Reading project! In this phase, our goal is to predict sentences from input video data without using audio information. This README provides a comprehensive overview of the project, including the dataset used, data preprocessing steps, deep neural network architecture, model training details, performance analysis, and individual contributions.

## Introduction

Lip Reading is a captivating project that delves into the realm of predicting sentences solely from video data. In this report, we present an overview of Phase 1, outlining the key components and progress made.

## Dataset

We leverage the Grid corpus, a dataset comprising 1000 short videos with corresponding alignment files for 34 individuals. The dataset includes diverse speakers, with 18 men and 16 women.

## Data Preprocessing

### Video Preprocessing
- Extracted 75 uniform frames from each video.
- Converted frames to grayscale.
- Manually cropped the mouth region from each frame for consistency.
- Standardized the data.

### Unique Character to Integer Encoding
- Mapped each character to an integer.
- Encoded vocabularies to their respective indices.
- Replaced characters not in the vocabulary with an empty string.

### Alignment Preprocessing
Alignments contain words corresponding to time stamps. Words are encoded similarly to unique character to integer encoding, considering silence as a space between words.

## Data Pipeline

Our data pipeline involves:
- Selecting 500 random videos from each folder.
- Allocating 450 videos for training and 50 for validation.
- Implementing the specified preprocessing steps.
- Incorporating prefetching for optimized performance.

## Deep Neural Network Architecture

Our current best model architecture consists of Convolutional Layers, Activation Layers (ReLU), MaxPooling Layers, TimeDistributed Flatten Layer, Bidirectional LSTM Layers, Dropout Layers, and a Dense Layer.

![Current Best Model](model.png)

## Model Training

- Utilizing the CTC loss function for effective training.
- Implementing a learning rate scheduler.
- Periodically saving the model to overcome Kaggle's time constraints.

## Model Performance

The model achieves over 90% accuracy in predicting sentences. Some examples:
- Epoch 1: Original - "bin white at t two now," Prediction - "le e e e e eo."
- Epoch 29: Original - "lay green in f one soon," Prediction - "lay gren in ne son."
- Epoch 50: Original - "bin blue by s six please," Prediction - "bin blue by six please."

![Training and Validation Loss Graph](loss.png)

## Tasks for Final Submission

- Train two additional models.
- Address the limitation of training on shorter sentences.
- Explore building a model for variable-length sentences.
- Conduct necessary accuracy analysis (word error rate and letter error rate).
- Develop a full-stack application to showcase the model's functionality.

## Individual Contribution

| Name   | Contributions                                          |
|--------|--------------------------------------------------------|
| Omm    | Manual crop proposal, dictionary mapping, modified CTC loss, Bi-GRU suggestion, model performance improvement. |
| Srihan | Auto crop proposal, learning rate scheduler, classic LSTM model proposal, model performance improvement.       |
| Nita   | 'base64' proposal, pretrained model suggestion, validation prediction function, model performance improvement. |

*Note: We collaborated using a single Kaggle account for convenience.*

## Conclusion

Phase 1 sets the foundation for the Lip Reading project, showing promising results. Stay tuned for the final submission, where we'll unveil further advancements and a complete GitHub repository.
