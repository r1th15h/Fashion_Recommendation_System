import sys
import json

from pipeline import search_products


if len(sys.argv) < 2:

    print(
        "Usage:"
    )

    print(
        "python src/test_pipeline.py <image_path> [max_price]"
    )

    sys.exit(1)


image_path = sys.argv[1]


max_price = None


if len(sys.argv) >= 3:

    max_price = float(
        sys.argv[2]
    )


result = search_products(
    image_path=image_path,
    max_price=max_price,
    final_k=10
)


print()
print(
    "========================================"
)

print(
    "QUERY INFORMATION"
)

print(
    "========================================"
)

print(
    "Category:",
    result["query"]["category"]
)

print(
    "Category confidence:",
    result[
        "query"
    ][
        "category_confidence"
    ]
)

print(
    "Color:",
    result["query"]["color"]
)

print(
    "Max price:",
    result["query"]["max_price"]
)


print()
print(
    "========================================"
)

print(
    "SEARCH RESULTS"
)

print(
    "========================================"
)


for i, product in enumerate(
    result["results"],
    start=1
):

    print()

    print(
        f"{i}. "
        f"{product['product_name']}"
    )

    print(
        f"   ID: {product['id']}"
    )

    print(
        f"   Category: "
        f"{product['category']}"
    )

    print(
        f"   Color: "
        f"{product['color']}"
    )

    print(
        f"   Price: "
        f"{product['price']}"
    )

    print(
        f"   Visual score: "
        f"{product['visual_score']:.4f}"
    )

    print(
        f"   Final score: "
        f"{product['final_score']:.4f}"
    )