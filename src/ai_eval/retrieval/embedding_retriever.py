import numpy as np
# This is a simple implementation of an embedding-based retriever. It assumes that the provider has an `embed` method that converts text into a vector representation. 
# The retriever computes the cosine similarity between the query embedding and the document embeddings to retrieve the most relevant documents.
class EmbeddingRetriever:
    def __init__(self, provider, documents):
        self.provider = provider
        self.documents = documents
        self.embeddings = [provider.embed(d) for d in documents]

    def retrieve(self, query, k=3):
        q = self.provider.embed(query)
        sims = [
            np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d))
            for d in self.embeddings
        ]
        return [self.documents[i] for i in sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]]
