#!pip install -U transformers tokenizers sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Load the free alternative model (Runs on CPU or GPU locally)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 2. Add the specific instruction prompt required by BGE models for queries
doc_chunk = "To reset your router, unplug the power cable, wait 10 seconds, and plug it back in."
user_query = "Represent this sentence for searching relevant passages: How do I restart my internet box?"

# 3. Generate the embeddings
doc_vector = model.encode(doc_chunk)
query_vector = model.encode(user_query)

# 4. Calculate similarity
similarity = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))

print(f"Vector Dimensions: {len(doc_vector)}")
print(f"Local Semantic Match Score: {similarity:.4f}")


#Paid One by Openai
# import numpy as np
# from openai import OpenAI

# # Initialize the OpenAI client (ensure OPENAI_API_KEY is in your environment)
# client = OpenAI()

# # Define your texts
# doc_chunk = "To reset your router, unplug the power cable, wait 10 seconds, and plug it back in."
# user_query = "How do I restart my internet box?"

# # 1. Embed the document chunk (Compressed to 1024 dimensions)
# doc_response = client.embeddings.create(
#     model="text-embedding-3-large",
#     input=doc_chunk,
#     dimensions=1024  # Cuts storage costs by 66% with minimal accuracy loss
# )
# doc_vector = doc_response.data[0].embedding

# # 2. Embed the user query (Must use the exact same dimensions)
# query_response = client.embeddings.create(
#     model="text-embedding-3-large",
#     input=user_query,
#     dimensions=1024
# )
# query_vector = query_response.data[0].embedding

# # 3. Calculate Cosine Similarity to find the match score
# dot_product = np.dot(query_vector, doc_vector)
# norm_query = np.linalg.norm(query_vector)
# norm_doc = np.linalg.norm(doc_vector)
# similarity_score = dot_product / (norm_query * norm_doc)

# print(f"Vector Dimensions: {len(doc_vector)}")
# print(f"Semantic Match Score: {similarity_score:.4f}")