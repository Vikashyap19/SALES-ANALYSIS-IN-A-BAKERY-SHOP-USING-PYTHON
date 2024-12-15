# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# Load the dataset
file_path = ""  # 
df = pd.read_csv(file_path)

# Basic Information about the dataset
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
df.info()
print("\nFirst 10 Rows of the Dataset:")
print(df.head(10))

# Check number of unique items in the 'article' column
print("\nNumber of unique items:", df['article'].nunique())
print("\nList of unique items:")
print(df['article'].unique())

# Check how many rows have 'NONE' as an item
none_count = (df['article'] == "NONE").sum()
print(f"\nNumber of rows with 'NONE' as an article: {none_count}")

# Visualize the 20 most sold items
print("\n20 Most Sold Items at the Shop:")
top_items = df['article'].value_counts().head(20)
plt.figure(figsize=(16, 7))
sns.barplot(x=top_items.index, y=top_items.values, palette="viridis")
plt.xlabel('Article')
plt.ylabel('Number of Transactions')
plt.title('20 Most Sold Items at the Shop')
plt.xticks(rotation=45)
plt.show()

# Prepare the dataset for Market Basket Analysis
# Step 1: Group data by transaction and article, then create a pivot table
print("\nPreparing data for Market Basket Analysis...")
hot_encoded_df = df.groupby(['ticket_number', 'article'])['article'].count().unstack().fillna(0)

# Step 2: Convert counts into binary format (0/1) for presence/absence of an item
hot_encoded_df = (hot_encoded_df > 0).astype(bool)
print("\nHot Encoded Data (First 5 rows):")
print(hot_encoded_df.head())

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.01  # Minimum support threshold
print("\nApplying Apriori Algorithm...")
frequent_itemsets = apriori(hot_encoded_df, min_support=min_support, use_colnames=True)
print("\nFrequent Itemsets (Top 10):")
print(frequent_itemsets.head(10))

# Generate Association Rules
min_lift = 1  # Minimum lift threshold
print("\nGenerating Association Rules...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)

# Sort rules by confidence
rules.sort_values('confidence', ascending=False, inplace=True)

# Display top 10 association rules
print("\nTop 10 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Visualize the association rules using a network graph
print("\nVisualizing Association Rules...")
G = nx.from_pandas_edgelist(
    rules, 'antecedents', 'consequents', 
    edge_attr=['support', 'confidence', 'lift']
)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(
    G, pos, with_labels=True, node_size=3000, node_color="lightblue", 
    font_size=10, font_color="black", edge_color="gray"
)
plt.title("Association Rules Network")
plt.show()
