# Next Word Prediction using Stacked Bidirectional LSTM + Multi-Head Attention

## Overview

This project implements a **Next Word Prediction Language Model** using a deep learning architecture based on:

- Stacked Bidirectional LSTM (Bi-LSTM)
- Multi-Head Self-Attention (MHA)
- TensorFlow / Keras
- Shakespeare Complete Works corpus

The model learns language patterns from Shakespeare's writings and predicts the most probable next word given a sequence of previous words.

---

## Project Objective

Build a language model capable of:

1. Learning grammar and contextual relationships between words.
2. Predicting the next word in a sentence.
3. Generating Shakespeare-style text.
4. Evaluating model quality using Accuracy, Top-K Accuracy, Cross-Entropy Loss, and Perplexity.

---

## Dataset

### Corpus
- Shakespeare Complete Works
- Downloaded from Project Gutenberg
- Includes plays, sonnets, and poems

### Source
https://www.gutenberg.org/ebooks/100

---

## Data Processing Pipeline

### 1. Corpus Loading
- Download and load Shakespeare corpus
- Combine text into a single corpus

### 2. Text Cleaning
The preprocessing pipeline performs:

- Lowercasing
- Removal of stage directions
- Removal of line numbers
- Removal of unwanted symbols
- Whitespace normalization

### 3. Tokenization
Keras Tokenizer is used to:

- Build vocabulary
- Create word-to-index mappings
- Create index-to-word mappings
- Handle out-of-vocabulary words

---

## Dataset Construction

The cleaned text is converted into training samples using a sliding window approach.

Example:

Input Sequence:

    to be or not to be that is the question

Target Word:

    whether

### Hyperparameters

| Parameter | Description |
|------------|-------------|
| SEQ_LEN | Context window length |
| STRIDE | Sliding window step size |
| VOCAB_SIZE | Vocabulary size |
| OOV | Out-of-vocabulary token |

---

## Model Architecture

Input
→ Embedding
→ SpatialDropout1D
→ BiLSTM (Layer 1)
→ BatchNormalization
→ Dropout
→ BiLSTM (Layer 2)
→ BatchNormalization
→ Dropout
→ BiLSTM (Layer 3)
→ Multi-Head Self-Attention
→ Residual Connection
→ Layer Normalization
→ GlobalAveragePooling1D
→ Dense (ReLU)
→ Dropout
→ Softmax Output

### Key Components

#### Embedding Layer
Transforms token IDs into dense vector representations.

#### Bidirectional LSTM
Captures contextual information from both forward and backward directions.

#### Multi-Head Self-Attention
Allows the model to learn relationships between distant words in a sequence.

#### Residual Connection
Preserves information learned by the LSTM layers while incorporating attention outputs.

#### Softmax Layer
Produces probability scores for every word in the vocabulary.

---

## Training Configuration

Typical training settings include:

- TensorFlow / Keras
- Adam Optimizer
- Categorical Crossentropy Loss
- Early Stopping
- Model Checkpointing
- Learning Rate Reduction
- Perplexity Monitoring

---

## Evaluation Metrics

### Accuracy
Percentage of exact next-word predictions.

### Top-5 Accuracy
Checks whether the correct word appears in the top five predictions.

### Cross Entropy Loss
Measures prediction error.

### Perplexity
Measures how well the model predicts the next word.

Lower perplexity indicates better language modeling performance.

---

## Visualizations

The notebook generates:

- Training Loss vs Validation Loss
- Training Accuracy vs Validation Accuracy
- Learning Curves
- Model Performance Summary

---

## Inference / Text Generation

The trained model can generate text by:

1. Taking a seed phrase.
2. Predicting the next word.
3. Appending the prediction.
4. Repeating the process iteratively.

Example:

    Input:
    "to be or not to"

    Predicted Next Word:
    "be"

---

## Project Structure

    next_word_prediction_lstm_v2.0.ipynb
    README.md
    complete_shakespeare.txt
    saved_model/
    outputs/

---

## Requirements

Install dependencies:

```bash
pip install tensorflow
pip install numpy pandas matplotlib
pip install nltk scikit-learn
```

---

## References

1. TensorFlow Text Generation Tutorial
   https://www.tensorflow.org/text/tutorials/text_generation

2. TensorFlow LSTM Documentation
   https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM

3. TensorFlow MultiHeadAttention Documentation
   https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention

4. Andrej Karpathy - The Unreasonable Effectiveness of Recurrent Neural Networks
   https://karpathy.github.io/2015/05/21/rnn-effectiveness/

5. Attention Is All You Need
   https://arxiv.org/abs/1706.03762

---

## Conclusion

This project demonstrates how modern sequence modeling techniques can be combined with traditional recurrent neural networks to build an effective next-word prediction system. By combining stacked Bidirectional LSTMs with Multi-Head Self-Attention, the model learns both sequential dependencies and long-range contextual relationships within Shakespearean text.
