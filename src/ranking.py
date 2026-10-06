def calculate_category_score(
    query_category,
    product_category
):

    if not query_category:
        return 0.0

    if not product_category:
        return 0.0

    query_category = str(
        query_category
    ).lower().strip()

    product_category = str(
        product_category
    ).lower().strip()


    # Exact match
    if query_category == product_category:
        return 1.0


    # Handle singular/plural differences
    if (
        query_category + "s"
        == product_category
    ):
        return 1.0

    if (
        product_category + "s"
        == query_category
    ):
        return 1.0


    # Handle categories such as:
    # "shoes" -> "casual shoes"
    # "shoes" -> "sports shoes"
    if query_category in product_category:
        return 1.0


    return 0.0


def calculate_color_score(
    query_color,
    product_color
):

    if not query_color:
        return 0.0

    if not product_color:
        return 0.0

    query_color = str(
        query_color
    ).lower().strip()

    product_color = str(
        product_color
    ).lower().strip()


    if query_color == product_color:
        return 1.0


    return 0.0


def calculate_price_score(
    product_price,
    max_price
):

    if max_price is None:
        return 0.0

    if product_price is None:
        return 0.0

    try:

        product_price = float(
            product_price
        )

        max_price = float(
            max_price
        )

    except (
        ValueError,
        TypeError
    ):

        return 0.0


    if max_price <= 0:
        return 0.0


    # Product is within budget
    if product_price <= max_price:
        return 1.0


    # Product is above budget
    difference = (
        product_price - max_price
    )

    score = 1.0 - (
        difference / max_price
    )

    return max(
        0.0,
        score
    )


def calculate_hybrid_score(
    visual_score,
    category_score,
    color_score,
    price_score,
    visual_weight=0.50,
    category_weight=0.35,
    color_weight=0.10,
    price_weight=0.05
):

    return (
        visual_weight * visual_score
        +
        category_weight * category_score
        +
        color_weight * color_score
        +
        price_weight * price_score
    )


def rerank_products(
    products,
    query_category=None,
    query_color=None,
    max_price=None,
    visual_weight=0.50,
    category_weight=0.35,
    color_weight=0.10,
    price_weight=0.05
):

    ranked_products = []


    for product in products:

        category_score = (
            calculate_category_score(
                query_category,
                product.get("category")
            )
        )


        color_score = (
            calculate_color_score(
                query_color,
                product.get("color")
            )
        )


        price_score = (
            calculate_price_score(
                product.get("price"),
                max_price
            )
        )


        visual_score = float(
            product.get(
                "visual_score",
                0.0
            )
        )


        final_score = (
            calculate_hybrid_score(
                visual_score,
                category_score,
                color_score,
                price_score,
                visual_weight,
                category_weight,
                color_weight,
                price_weight
            )
        )


        ranked_product = product.copy()


        ranked_product[
            "category_score"
        ] = category_score


        ranked_product[
            "color_score"
        ] = color_score


        ranked_product[
            "price_score"
        ] = price_score


        ranked_product[
            "final_score"
        ] = final_score


        ranked_products.append(
            ranked_product
        )


    ranked_products.sort(
        key=lambda x: x[
            "final_score"
        ],
        reverse=True
    )


    return ranked_products