
# CSE665: Large Language Models - Assignment 1


## Overview

This repository contains the implementation for CSE665 Assignment 1, which explores the capabilities and limitations of Large Language Models (LLMs) like LLAMA 3.1 and OpenHathi. The primary objectives of this assignment were to analyze the performance of these models in handling hallucinations and in encoding knowledge for various tasks like regression and classification using different layers of the model.


## Contents

- **Main Jupyter Notebook:**
  - `CSE665_Assignment1_2021041.ipynb`: This is the main Jupyter notebook containing the code for extracting embeddings, training linear regression and classification models, and analyzing hallucinations in LLMs.

- **Scripts:**
  - `cse665_assignment1_2021041.py`: Python script containing modularized code for the extraction of embeddings and implementation of classification/regression tasks.

- **PDF Report:**
  - `Large Language Models A1.pdf`: Detailed report discussing the findings and analysis from the experiments.



## Datasets

- **IMDB Top 1000 Movies**:
  Used for the linear regression task to predict IMDB ratings based on movie descriptions.

- **DBpedia 14**:
  Used for the classification task to predict the entity class (e.g., person, place, organization) based on text descriptions.



## Key Experiments

### Part 1: Hallucination Analysis

We classified hallucinations in the LLM responses into two categories:
- **Factual Hallucinations**: Incorrect or misleading factual content.
- **Faithfulness Hallucinations**: Unfaithful information, such as adding irrelevant or unnecessary content.

**Retrieval-Augmented Generation (RAG):**
RAG was implemented to reduce hallucinations by using external knowledge from a Pinecone vector store to improve factual consistency.

### Part 2: Probing Tasks (Regression and Classification)

- **Embedding Extraction**: 
  - Embeddings from different layers (first, middle, final) of LLAMA 3.1 and OpenHathi were extracted.
  
- **Regression on IMDB Dataset**:
  - A linear regression model was trained to predict IMDB ratings based on embeddings.

- **Classification on DBpedia 14**:
  - A Random Forest classifier was trained to predict entity class labels based on extracted embeddings.

### Part 3: Analysis

We compared the performance across different model layers:
- **Middle layers** generally provided the best balance of general-purpose and task-specific information, leading to superior performance for both regression and classification tasks.
- **First layers** capture surface-level features and performed poorly.
- **Final layers** were highly task-specific, showing a slight decline in general-purpose tasks.



## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/CSE665_LLM_Assignment1.git
cd CSE665_LLM_Assignment1
```

2. **Run the Jupyter Notebook:**

Open `CSE665_Assignment1_2021041.ipynb` in a Jupyter environment to reproduce the experiments.

3. **Required Dependencies:**

- Python 3.8 or higher
- Hugging Face Transformers
- Pinecone
- Scikit-learn
- NumPy

To install the required dependencies:

```bash
pip install -r requirements.txt
```

4. **Run the Python script:**

To run the main Python script `cse665_assignment1_2021041.py`, use:

```bash
python cse665_assignment1_2021041.py
```


## Conclusion

This project demonstrates how large language models (LLMs) encode information at different layers and their effectiveness across various tasks. By using techniques such as RAG, hallucinations were reduced, and factual consistency was improved.
