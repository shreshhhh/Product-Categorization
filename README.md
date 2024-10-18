# Product Classification and Recommendation System 📊🛍️

## Overview
This project focuses on two key functionalities:

1. **Product Classification**: Classifies products into subcategories based on their descriptions using Logistic Regression and TF-IDF vectorization.
2. **Similar Product Recommendation**: Recommends products similar to a given product by leveraging Sentence Embeddings (via Hugging Face's SentenceTransformer) and Cosine Similarity.

---

## Features

### 1. Product Classification
- **TF-IDF Vectorization**: Converts product descriptions into numerical vectors.
- **Logistic Regression Model**: Trained to classify products into subcategories.
- **Model Evaluation**: Assessed using a confusion matrix and classification report.
- **Tabulated Results**: Displays predictions in a structured, readable table using the `tabulate` library.

### 2. Similar Product Recommendation
- **Sentence Embeddings**: Uses Hugging Face's `SentenceTransformer` model to convert product descriptions into sentence embeddings.
- **Cosine Similarity**: Calculates the similarity between products based on their descriptions and recommends the most similar ones.

---


## Usage

### 1. Product Classification

#### Step-by-step guide:

1. **Load Dataset**:
   Load the product dataset (CSV format) and preprocess it by removing rows with missing values.

2. **TF-IDF Vectorization**:
   Convert product descriptions into numerical features using TF-IDF vectorization.

3. **Model Training**:
   Train a Logistic Regression model to classify products into subcategories.

4. **Model Evaluation**:
   Evaluate the model using confusion matrices, classification reports, and tabulated results.

---

### 2. Similar Product Recommendation

#### Step-by-Step Guide:

1. **Load Dataset**:  
   Load the product dataset and extract necessary columns (product name and description).

2. **Generate Embeddings**:  
   Use Hugging Face's SentenceTransformer model to generate sentence embeddings from product descriptions.

3. **Cosine Similarity**:  
   Compute the similarity between product descriptions and recommend the most similar products.

4. **Output Recommendations**:  
   Provide a list of recommended products with similarity scores.

---


## Prerequisites
- **Python 3.x**
- **pip** (Python package installer)
- **scikit-learn**
- **sentence-transformers**
- **pandas**
- **matplotlib**
- **seaborn**
- **tabulate**

  
## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/product-classification-recommendation.git
    cd product-classification-recommendation
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run Product Classification**:
    ```bash
    python Task1/product_classification.py
    ```

4. **Run Product Recommendation**:
    ```bash
    python Task2/product_recommendation.py
    ```

---

## Results and Evaluation

### 1. Classification Results:
- The classification model's predictions are displayed in a tabulated format with columns for:
   - **Product Name**
   - **Predicted Subcategory**
   - **True Subcategory**

- The model is evaluated using a **Confusion Matrix** and a detailed **Classification Report** (precision, recall, F1-score).

### 2. Product Recommendation Results:
- Given a product, the system recommends the top `n` most similar products based on cosine similarity of their embeddings.

---

## Contact
**Developer**: Shrestha  
Feel free to reach out at: shresthaaa16@gmail.com
