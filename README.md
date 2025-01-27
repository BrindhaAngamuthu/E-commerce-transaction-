import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Convert date columns
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# EDA: Number of Customers per Region
plt.figure(figsize=(8,5))
sns.countplot(data=customers, x='Region', order=customers['Region'].value_counts().index)
plt.title("Number of Customers per Region")
plt.xticks(rotation=45)
plt.show()

# EDA: Top 10 Best-Selling Products
top_products = transactions['ProductID'].value_counts().head(10)
plt.figure(figsize=(8,5))
sns.barplot(x=top_products.index, y=top_products.values)
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Product ID")
plt.ylabel("Quantity Sold")
plt.show()

# Merge transactions with customer and product data
merged_df = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Feature Engineering for Lookalike Model
customer_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique'
}).reset_index()

# Standardize features
scaler = StandardScaler()
customer_features_scaled = scaler.fit_transform(customer_features.iloc[:, 1:])

# Compute similarity matrix
similarity_matrix = cosine_similarity(customer_features_scaled)

# Lookalike Recommendations (Top 3 Similar Customers)
customer_ids = customer_features['CustomerID'].tolist()
lookalike_results = {}

for idx, customer in enumerate(customer_ids[:20]):  # First 20 customers
    similar_customers = sorted(
        list(enumerate(similarity_matrix[idx])), 
        key=lambda x: x[1], reverse=True)[1:4]
    
    lookalike_results[customer] = [(customer_ids[i], round(score, 2)) for i, score in similar_customers]

# Save Lookalike Results
lookalike_df = pd.DataFrame(list(lookalike_results.items()), columns=['CustomerID', 'Lookalikes'])
lookalike_df.to_csv("Lookalike.csv", index=False)
print("Lookalike.csv saved successfully!")

# Clustering: Apply K-Means
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans.fit_predict(customer_features_scaled)

