import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Loading dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} rows.")
    
    # Selecting necessarycolumns
    data = data[['Description', 'SubCategory', 'ProductName']]
    
    # Removing rows with missing values 
    data.dropna(subset=['Description', 'SubCategory'], inplace=True)
    
    print(f"Remaining rows after dropping NA: {len(data)}")
    return data

# TF-IDF Vectorization
#  converting product descriptions into numerical values using a TF-IDF,
def vectorize_data(data):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(data['Description'])
    return X, vectorizer

# Training Model 
def train_model(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Model Evaluation and Displaying Results & confusion matrix
def evaluate_model(model, X_test, y_test, product_names_test):
    y_pred = model.predict(X_test)
    
    # Creating DataFrame with product names, predicted subcategorie and true subcategories
    results = pd.DataFrame({
        'ProductName': product_names_test,
        'Predicted_SubCategory': y_pred,
        'True_SubCategory': y_test
    })
    
    # Displaying 100 rows we can change number 
    # using tabulate to make table and give structure
    print(tabulate(results.head(100), headers='keys', tablefmt='grid'))
    
    # confusion matrix  to evaluate performance
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plotting confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted SubCategory')
    plt.ylabel('True SubCategory')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Classification report
    print(classification_report(y_test, y_pred, target_names=model.classes_))
    
    return results

# Main Pipeline Execution
if __name__ == "__main__":
    # Loading dataset
    data = load_data('./Task1/dataset.csv')
    
    # Vectorizing the product descriptions
    X, vectorizer = vectorize_data(data)
    
    # Define target labels as subcategories
    y = data['SubCategory']
    
    # Splitting the data into training and test.
    X_train, X_test, y_train, y_test, product_names_train, product_names_test = train_test_split(
        X, y, data['ProductName'], test_size=0.2, random_state=42)
    
    # Train using logistic regression model
    model = train_model(X_train, y_train)
    
    # Evaluating the model and displaying the classification results and confusion matrix
    results = evaluate_model(model, X_test, y_test, product_names_test)
    
    # Saving results.
    results.to_csv('results.csv', index=False)
