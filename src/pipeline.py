from PIL import Image

from classifier import predict_category
from color import predict_color
from search import retrieve_products
from ranking import rerank_products


def search_products(
    image_path,
    max_price=None,
    final_k=10
):

    # ========================================================
    # LOAD QUERY IMAGE
    # ========================================================

    image = Image.open(
        image_path
    ).convert("RGB")


    # ========================================================
    # PREDICT CATEGORY
    # ========================================================

    query_category, category_confidence = (
        predict_category(
            image
        )
    )


    # ========================================================
    # DETECT COLOR
    # ========================================================

    query_color, color_confidence = (
        predict_color(
            image
        )
    )


    # ========================================================
    # VISUAL RETRIEVAL
    # ========================================================

    candidates = retrieve_products(
        image_path,
        k=50
    )


    # ========================================================
    # HYBRID RE-RANKING
    # ========================================================

    ranked_results = rerank_products(
        candidates,
        query_category=query_category,
        query_color=query_color,
        max_price=max_price
    )


    # ========================================================
    # RETURN FINAL RESULTS
    # ========================================================

    return {
        "query": {
            "category": query_category,
            "category_confidence": category_confidence,
            "color": query_color,
            "color_confidence": color_confidence,
            "max_price": max_price
        },

        "results": ranked_results[:final_k]
    }