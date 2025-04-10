import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import os
import pickle

# Specify Cache Names (For faster subsequent tests)
EMBEDDINGS_CACHE = "job_embeddings.pkl"
FAISS_INDEX_CACHE = "faiss_index.faiss"
DF_CACHE = "job_dataframe.pkl"


def load_or_create_resources():
    # Check if cache exists
    if all(os.path.exists(f) for f in [EMBEDDINGS_CACHE, FAISS_INDEX_CACHE, DF_CACHE]):
        print("Cache Found... Loading!")
        with open(DF_CACHE, 'rb') as f:
            df = pickle.load(f)
        with open(EMBEDDINGS_CACHE, 'rb') as f:
            embeddings = pickle.load(f)
        index = faiss.read_index(FAISS_INDEX_CACHE)

    else:
        print("No Cache Found... Creating now (be patient!)")
        # Load and prepare data
        df = pd.read_csv("soc2020volume2thecodingindexexcel16102024.csv", encoding="ISO-8859-1")
        df['combined_text'] = (
                df['SOC2020_ext_SUG_title'].str.lower() + " " +
                df['INDEXOCC_-_natural_word_order'].str.lower()
        )

        # Specify Model (JobBert)
        model = SentenceTransformer('TechWolf/JobBERT-v2')

        # Create Embeddings from SOC Data
        embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)
        embeddings = normalize(embeddings, norm='l2', axis=1)

        # Create Index from Embeddings for Faster Searching
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))

        # Save Index and Embeddings
        with open(DF_CACHE, 'wb') as f:
            pickle.dump(df, f)
        with open(EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings, f)
        faiss.write_index(index, FAISS_INDEX_CACHE)

    return df, embeddings, index


# Load data and resources (either from cache or create new)
df, embeddings, index = load_or_create_resources()


def get_job_code(query, top_k=1):
    """Get the best matching SOC code for a free-text job title query"""

    model = SentenceTransformer('TechWolf/JobBERT-v2')

    # Take query and create embedding to search with
    query_embedding = model.encode([query.lower()])
    query_embedding = normalize(query_embedding, norm='l2', axis=1)

    # Find most similar embedding in the index to the query embedding
    distances, indices = index.search(query_embedding.astype(np.float32), top_k)

    # Return top match
    match = df.iloc[indices[0][0]]
    return {
        'SOC_Code': match['SOC_2020_ext'],
        'SOC_Title': match['SOC2020_ext_SUG_title'],
        'natural_word_order': match['INDEXOCC_-_natural_word_order'],
        'Similarity_Score': distances[0][0]
    }


# Test cases
queries = [
    "Labourer Forest",
    "Labourer",
    "Shelf Stacker",
    "Shop Worker",
    "Engineer",
    "Warehouse worker",
    "Data scientist",
    "Hospital cleaner",
    "Garbage collector",
    "SDET",
    "Developer in Test",
    "Data Warehouse",
    "Software Developer",
    "Java Engineer",
    "Java Developer",
    "Javascript Developer"
]
for query in queries:
    result = get_job_code(query)
    print(f"You queried: {query}")
    print(f"Best Match: {result['SOC_Code']} - {result['SOC_Title']}")
    print(f"Natural Word Order: {result['natural_word_order']}")
    print(f"Similarity: {result['Similarity_Score']:.3f}\n")
