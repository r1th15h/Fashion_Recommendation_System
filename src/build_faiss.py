import os
import numpy as np
import faiss


EMBEDDINGS_PATH = "embeddings/product_embeddings.npy"
INDEX_PATH = "index/fashion.index"


def build_faiss_index():

    embeddings = np.load(
        EMBEDDINGS_PATH
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    os.makedirs(
        "index",
        exist_ok=True
    )

    faiss.write_index(
        index,
        INDEX_PATH
    )

    print()
    print("FAISS index created successfully!")
    print("Number of vectors:", index.ntotal)
    print("Embedding dimension:", dimension)
    print("Saved to:", INDEX_PATH)


if __name__ == "__main__":
    build_faiss_index()