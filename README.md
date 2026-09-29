# Multiclass Text Classification using TF-IDF, PyTorch MLP & Word2Vec

A complete NLP text-classification pipeline comparing **traditional TF-IDF features** with **pretrained Word2Vec embeddings** for multiclass news-category classification.

The project implements classical machine learning and neural-network approaches using **Scikit-learn and PyTorch**, including Logistic Regression, PyTorch Logistic Regression, and Multi-Layer Perceptrons (MLPs). It also investigates optimizers, activation functions, network depth, and hyperparameter configurations.

---

## 📌 Project Overview

The goal of this project is to investigate how different text representations and machine-learning architectures affect multiclass text-classification performance.

The project follows this progression:

```text
Raw Text
   ↓
Text Cleaning & Preprocessing
   ↓
Train / Validation / Test Split
   ↓
 ┌───────────────────────────────┐
 │                               │
 ▼                               ▼
TF-IDF                         Word2Vec
 │                               │
 ▼                               ▼
Logistic Regression             Average Pooling
 │                               │
 ▼                               ▼
PyTorch MLP ──────────────────► PyTorch MLP
 │                               │
 └──────────────┬────────────────┘
                ▼
        Model Evaluation
                ↓
 Accuracy / Precision / Recall / F1
```

---

## 🎯 Objectives

* Build a complete multiclass NLP classification pipeline.
* Perform text cleaning and preprocessing.
* Analyze category distributions and text characteristics.
* Implement **TF-IDF** feature extraction.
* Train and evaluate Logistic Regression.
* Implement Logistic Regression using PyTorch.
* Build configurable PyTorch MLP models.
* Compare **SGD vs Adam** optimizers.
* Compare different activation functions.
* Investigate different MLP depths.
* Perform hyperparameter experimentation.
* Build sentence embeddings using pretrained **Word2Vec**.
* Train an MLP using Word2Vec sentence representations.
* Compare sparse lexical features against dense semantic embeddings.
* Evaluate models using Accuracy, Precision, Recall, F1-score, and confusion matrices.

---

## 📊 Dataset

The project uses the **Exploration-Lab/CS781** dataset loaded through the Hugging Face `datasets` library.

For computational efficiency, a sample of **10,000 training samples** was used.

### Dataset split

| Dataset    | Samples |
| ---------- | ------: |
| Training   |   8,000 |
| Validation |   2,000 |
| Test       |   7,600 |

The 10,000 training samples were split using an **80/20 stratified split**.

### Classes

The classification task contains four categories:

* Business
* Sci/Tech
* Sports
* World

The sampled training dataset is relatively balanced across these four categories.

---

## 🧹 Text Preprocessing

The raw `Description` field is cleaned using a custom preprocessing function.

The preprocessing pipeline includes:

1. Convert text to string.
2. Convert text to lowercase.
3. Remove URLs.
4. Remove HTML tags.
5. Remove punctuation.
6. Normalize whitespace.
7. Remove missing/empty descriptions.

Example:

```text
Raw Description
      ↓
Lowercase
      ↓
Remove URLs
      ↓
Remove HTML
      ↓
Remove punctuation
      ↓
Normalize whitespace
      ↓
Clean Description
```

---

# 🔹 Part 1 — TF-IDF Representation

TF-IDF is used as the primary sparse lexical representation.

The project uses:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2
)
```

Therefore, the feature representation contains up to **10,000 features** using:

* Unigrams
* Bigrams
* Minimum document frequency = 2

The resulting shapes were:

```text
Training:   (8000, 10000)
Validation: (2000, 10000)
Test:       (7600, 10000)
```

---

# 🔹 Part 2 — Logistic Regression

Logistic Regression is used as a strong classical baseline.

### Scikit-learn Logistic Regression

The model achieved on the validation set:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **87.70%** |
| Precision | **87.71%** |
| Recall    | **87.70%** |
| F1-score  | **87.63%** |

This provides a useful baseline before moving to neural-network models.

---

# 🔹 Part 3 — PyTorch Logistic Regression

A Logistic Regression-style model was also implemented using **PyTorch**.

This allows the project to introduce:

* PyTorch tensors
* Neural-network modules
* Loss functions
* Optimizers
* Backpropagation
* Training loops

The PyTorch implementation achieved:

| Metric   |      Score |
| -------- | ---------: |
| Accuracy | **80.20%** |
| F1-score | **79.76%** |

---

# 🔹 Part 4 — Multi-Layer Perceptron

A configurable **Multi-Layer Perceptron (MLP)** was implemented using PyTorch.

General architecture:

```text
Input Features
      ↓
Linear Layer
      ↓
Activation Function
      ↓
Hidden Layer(s)
      ↓
Output Layer
      ↓
Class Prediction
```

The MLP implementation supports configurable:

* Input dimension
* Hidden-layer dimensions
* Number of classes
* Activation functions

The project uses:

```text
CrossEntropyLoss
```

for multiclass classification.

---

# 🔹 Part 5 — Activation Function Experiment

Five activation functions were compared:

* Sigmoid
* ReLU
* Tanh
* Leaky ReLU
* GELU

Validation results:

| Activation |   Accuracy |         F1 |
| ---------- | ---------: | ---------: |
| Sigmoid    | **88.50%** | **88.47%** |
| ReLU       |     87.45% |     87.44% |
| Tanh       |     87.20% |     87.18% |
| Leaky ReLU |     87.20% |     87.19% |
| GELU       |     87.20% |     87.18% |

The experiment demonstrates how changing only the activation function can affect classification performance.

---

# 🔹 Part 6 — MLP Architecture Experiment

Different numbers of hidden layers were investigated.

### Architecture 1

```text
Input
 ↓
128
 ↓
Output
```

### Architecture 2

```text
Input
 ↓
128
 ↓
64
 ↓
Output
```

### Architecture 3

```text
Input
 ↓
256
 ↓
128
 ↓
64
 ↓
Output
```

Validation results:

| Architecture    |   Accuracy |         F1 |
| --------------- | ---------: | ---------: |
| 1 Hidden Layer  | **87.25%** | **87.24%** |
| 2 Hidden Layers |     86.90% |     86.89% |
| 3 Hidden Layers |     85.90% |     85.90% |

The experiments also illustrate that increasing network depth does not automatically improve validation performance.

---

# 🔹 Part 7 — Hyperparameter Experimentation

The project explores combinations of:

* Learning rate
* Hidden dimension
* Activation function
* Optimizer

The tested learning rates include:

```text
0.01
0.001
```

Hidden dimensions:

```text
64
128
```

Activations:

```text
ReLU
GELU
```

Optimizers:

```text
Adam
SGD
```

A total of **16 configurations** were evaluated.

The highest validation result in this experiment was:

```text
Learning Rate : 0.001
Hidden Size   : 64
Activation    : GELU
Optimizer     : Adam
Accuracy      : 87.60%
F1 Score      : 87.60%
```

---

# 🔹 Part 8 — Pretrained Word2Vec + MLP

The final stage replaces sparse TF-IDF features with dense pretrained Word2Vec embeddings.

The pretrained model used is:

```text
word2vec-google-news-300
```

Each word is represented by a **300-dimensional vector**.

### Sentence representation

A sentence is converted into one fixed-size vector by averaging the Word2Vec vectors of its in-vocabulary words.

```text
Sentence
   ↓
Tokenization
   ↓
Word2Vec lookup
   ↓
Word vectors
   ↓
Average pooling
   ↓
300-dimensional sentence vector
   ↓
MLP
   ↓
Class
```

Out-of-vocabulary words are skipped.

If a sentence contains no in-vocabulary words, a zero vector is used.

---

## 🧠 Word2Vec MLP Architecture

The trained Word2Vec MLP uses:

```text
Input Dimension: 300

300
 ↓
128
 ↓
ReLU
 ↓
4 Classes
```

Equivalent PyTorch architecture:

```python
MLP(
    input_dim=300,
    hidden_dims=[128],
    num_classes=4,
    activation=nn.ReLU
)
```

The model is trained using:

```text
Optimizer: Adam
Learning Rate: 0.001
Activation: ReLU
Hidden Layer: 128 neurons
Embedding: Word2Vec Google News 300
```

---

# 📈 Final Word2Vec Results

The Word2Vec + MLP model achieved the following validation performance:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **88.95%** |
| Precision | **88.92%** |
| Recall    | **88.95%** |
| F1-score  | **88.92%** |

### Classification Report

| Category | Precision | Recall |   F1 |
| -------- | --------: | -----: | ---: |
| Business |      0.86 |   0.83 | 0.85 |
| Sci/Tech |      0.85 |   0.87 | 0.86 |
| Sports   |      0.93 |   0.97 | 0.95 |
| World    |      0.91 |   0.88 | 0.90 |

Validation set size:

```text
2,000 samples
```

---

# 📊 Model Comparison

The main model comparison from the project is:

| Model               | Framework    | Features            |   Accuracy |         F1 |
| ------------------- | ------------ | ------------------- | ---------: | ---------: |
| Logistic Regression | Scikit-learn | TF-IDF              |     87.70% |     87.63% |
| Logistic Regression | PyTorch      | TF-IDF              |     80.20% |     79.76% |
| MLP                 | PyTorch      | TF-IDF              |     87.35% |     87.33% |
| MLP                 | PyTorch      | Pretrained Word2Vec | **88.95%** | **88.92%** |

These results provide an experimental comparison between:

```text
Sparse lexical representation
        vs
Dense semantic representation
```

---

# 🔍 Evaluation Metrics

The project evaluates classification models using:

### Accuracy

Percentage of correctly classified samples.

### Precision

Measures how many samples predicted as a particular class were actually from that class.

### Recall

Measures how many samples belonging to a class were correctly identified.

### F1-score

Harmonic mean of precision and recall.

### Confusion Matrix

Used to analyze which categories are being confused with one another.

---

# ⚙️ Training Pipeline

The PyTorch training process follows the standard workflow:

```text
Input Batch
    ↓
Forward Pass
    ↓
Predictions
    ↓
Cross Entropy Loss
    ↓
Backpropagation
    ↓
Optimizer Step
    ↓
Updated Weights
```

Validation is performed using:

```python
model.eval()
torch.no_grad()
```

Training and validation losses are tracked to analyze model learning and potential overfitting.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* PyTorch

### NLP

* TF-IDF
* Word2Vec
* Tokenization
* Text preprocessing

### Data Processing

* NumPy
* Pandas
* Hugging Face Datasets

### Visualization

* Matplotlib
* Seaborn

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch datasets huggingface_hub gensim
```

---

# 🚀 Running the Project

Launch the Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Multiclass Text Classification using TF-IDF pytorch MLP and Word2Vec Tejavath Bhaskar.ipynb
```

Run the notebook sequentially.

> **Note:** The pretrained `word2vec-google-news-300` model is large and requires additional download/storage space when first loaded through Gensim.

---

# 💾 Model Checkpoint

The trained Word2Vec MLP is saved as:

```text
221138_word2vec_mlp.pt
```

The checkpoint contains:

```python
{
    "student_id": ...,
    "seed": ...,
    "timestamp": ...,
    "model_config": ...,
    "model_state_dict": ...,
    "label_classes": ...,
    "embedding_config": ...
}
```

The saved model can be reconstructed using the stored architecture and activation configuration.

---

# 🔬 Key Learnings

### TF-IDF

* Sparse representation
* Vocabulary-based
* Computationally efficient
* Strong traditional baseline
* Does not directly encode semantic relationships between words

### Word2Vec

* Dense representation
* Fixed 300-dimensional embeddings
* Captures semantic relationships learned from a large pretrained corpus
* Sentence representations are created using average pooling
* Average pooling does not preserve word order

### Logistic Regression

* Simple linear baseline
* Fast to train
* Useful for evaluating the quality of the feature representation

### MLP

* Learns nonlinear decision boundaries
* More expressive than a linear classifier
* Allows experimentation with architecture and activation functions

### Adam vs SGD

The experiments demonstrate the importance of optimizer and learning-rate selection when training neural networks.

### Hyperparameter Tuning

The project demonstrates experimentation with:

```text
Learning Rate
Hidden Dimensions
Activation Function
Optimizer
Network Depth
```

---

# 📁 Suggested Repository Structure

```text
multiclass-text-classification/
│
├── README.md
│
├── notebooks/
│   └── Multiclass Text Classification using TF-IDF pytorch MLP and Word2Vec Tejavath Bhaskar.ipynb
│
├── models/
│   └── 221138_word2vec_mlp.pt
│
├── results/
│   ├── confusion_matrix.png
│   └── model_comparison.csv
│
├── requirements.txt
│
└── .gitignore
```

---

# 👨‍💻 Author

**Tejavath Bhaskar**

Materials Science & Engineering
Indian Institute of Technology Kanpur

Interests:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Generative AI
* Computer Vision
* Data Science

---

# 📚 References

* Scikit-learn Documentation — https://scikit-learn.org/
* PyTorch Documentation — https://pytorch.org/
* Gensim Documentation — https://radimrehurek.com/gensim/
* Hugging Face Datasets — https://huggingface.co/docs/datasets/
* Word2Vec — Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*

---

## ⭐ Project Highlights

```text
✔ Multiclass NLP Classification
✔ TF-IDF Feature Engineering
✔ Logistic Regression Baseline
✔ PyTorch Neural Networks
✔ MLP Architecture Experiments
✔ Activation Function Comparison
✔ SGD vs Adam
✔ Hyperparameter Search
✔ Pretrained Word2Vec Embeddings
✔ Sentence-Level Embedding via Average Pooling
✔ Confusion Matrix Analysis
✔ Model Checkpointing
✔ 88.95% Validation Accuracy
✔ 88.92% Validation F1-score
```
