import pandas as pd

df = pd.read_excel('dataset/raw/Search Queries - July 2025.xlsx')

print(df.columns)

tmp_df = df[['Search Queries Dump', 'Query Type']]
tmp_df = tmp_df.dropna(subset=['Query Type'])
print(f"Unique label values: {tmp_df['Query Type'].unique().tolist()}")
print(f"Number of samples: {len(tmp_df)}")
print(f"Count of Outfit: {tmp_df['Query Type'].value_counts().get('Outfit Idea Query', 0)}")
print(f"Count of Product: {tmp_df['Query Type'].value_counts().get('Product Query', 0)}")
print(f"Count of Grape Prompted Query: {tmp_df['Query Type'].value_counts().get('Grape Prompted Query', 0)}")
print(f"Count of Mixed: {tmp_df['Query Type'].value_counts().get('Mixed', 0)}")

# keep only outfit and product rows
# filtered_df = tmp_df[tmp_df['Query Type'].isin(['Outfit Idea Query', 'Product Query'])]
# print(f"Number of samples after filtering: {len(filtered_df)}")

# filtered_df.to_csv('dataset/processed/search_queries_july_2025.csv', index=False)
