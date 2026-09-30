import numpy as np
import pandas as pd
import faiss

from PIL import Image

from clip_encoder import get_image_embedding

from config import (
    FAISS_INDEX,
    PRODUCT_IDS,
    PRODUCTS_CSV
)


def retrieve_products(
    image_path,
    k=50
):

    # --------------------------------------------------------
    # Load FAISS index
    # --------------------------------------------------------

    index = faiss.read_index(
        FAISS_INDEX
    )


    # --------------------------------------------------------
    # Load product IDs
    # --------------------------------------------------------

    product_ids = np.load(
        PRODUCT_IDS
    )


    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    products = pd.read_csv(
        PRODUCTS_CSV
    )


    # --------------------------------------------------------
    # Load query image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = (
        get_image_embedding(
            image
        )
    ).astype(
        "float32"
    )


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        k
    )


    results = []


    # --------------------------------------------------------
    # Convert FAISS results to products
    # --------------------------------------------------------

    for score, position in zip(
        scores[0],
        indices[0]
    ):

        if position < 0:
            continue


        product_id = int(
            product_ids[position]
        )


        matching_products = products[
            products["id"] == product_id
        ]


        if matching_products.empty:
            continue


        product = (
            matching_products.iloc[0]
        )


        price = product.get(
            "discounted_price"
        )


        if pd.isna(price):
            price = product.get(
                "price"
            )


        results.append(
            {
                "id": product_id,

                "image_path":
                    product[
                        "image_path"
                    ],

                "category":
                    product[
                        "articleType"
                    ],

                "color":
                    product[
                        "baseColour"
                    ],

                "price":
                    price,

                "product_name":
                    product[
                        "productDisplayName"
                    ],

                "visual_score":
                    float(score)
            }
        )


    return results

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print("Usage:")
        print("python src/search.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    results = retrieve_products(
        image_path,
        k=10
    )

    print()
    print("========================================")
    print("SEARCH RESULTS")
    print("========================================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print()
        print(f"Rank: {i}")
        print("Product ID:", result["id"])
        print("Name:", result["product_name"])
        print("Category:", result["category"])
        print("Color:", result["color"])
        print("Price:", result["price"])
        print("Visual Score:", result["visual_score"])
        print("Image:", result["image_path"])