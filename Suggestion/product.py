import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Loading dataset
def load_data(file_path):
    data = pd.read_csv(file_path) 
    data = data[['ProductName', 'Description']].dropna(subset=['Description']) #Taking necessary columns
    return data.reset_index(drop=True)  

# Generating text embeddings using Hugging Face model
def generate_embeddings(data, model_name='sentence-transformers/all-MiniLM-L6-v2'): #using open-source pre-trained language model
    model = SentenceTransformer(model_name)
    descriptions = data['Description'].tolist()
    embeddings = model.encode(descriptions, show_progress_bar=True)
    return embeddings

# Recommeding similar products
def recommend_similar_products(product_name, data, embeddings, top_n=5):
    # Checking if the product exists in the dataset
    matching_products = data[data['ProductName'].str.strip() == product_name.strip()]

    if matching_products.empty:
        raise ValueError(f"Product '{product_name}' not found in the dataset.")
    
    # Getting the index of the target product
    product_idx = matching_products.index[0]
    
    # Computing cosine similarities between the target product and all others
    target_embedding = embeddings[product_idx].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, embeddings).flatten()
    
    # Getting the indices of the top N most similar products (excluding the product itself)
    similar_indices = similarities.argsort()[-(top_n + 1):-1][::-1]
    
    # Retrieving the names and similarity scores of the recommended products
    recommended_products = data.iloc[similar_indices].copy()
    recommended_products['Similarity'] = similarities[similar_indices]
    
    return recommended_products[['ProductName', 'Similarity']]

#Main functio
if __name__ == "__main__":
    try:
        # # Loading dataset
        data = load_data('./Task2/dataset.csv')

        #  unique products for debugging
        unique_products = data['ProductName'].unique()
        print("Available products:")
        for product in unique_products[:5]:  # Getting only first 5 products || we can change values
            print(product)

        # Generating embeddings for product descriptions
        embeddings = generate_embeddings(data)

        # Recommending similar products for a specific product
        product_name = input("Enter the product name to find similar products: ")
        recommended = recommend_similar_products(product_name, data, embeddings)

        print("\nRecommended similar products:\n", recommended)

    except ValueError as e:
        print(e)

