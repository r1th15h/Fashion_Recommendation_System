import os
import csv
import json

import pandas as pd

from config import (
    DATASET_DIR,
    IMAGES_CSV,
    STYLES_CSV,
    STYLES_DIR,
    DATA_DIR,
    PRODUCTS_CSV
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# READ IMAGES CSV
# ============================================================

print("Reading images.csv...")

images_df = pd.read_csv(
    IMAGES_CSV
)

images_df["id"] = (
    images_df["filename"]
    .str.replace(".jpg", "", regex=False)
    .astype(int)
)

images_df = images_df[
    [
        "id",
        "filename",
        "link"
    ]
]


# ============================================================
# READ STYLES CSV
# ============================================================

print("Reading styles.csv...")

rows = []

with open(
    STYLES_CSV,
    "r",
    encoding="utf-8",
    errors="replace"
) as f:

    reader = csv.reader(f)

    header = next(reader)

    for row in reader:

        if len(row) < 10:
            continue

        row = row[:9] + [
            ",".join(row[9:])
        ]

        rows.append(row)


styles_df = pd.DataFrame(
    rows,
    columns=header
)


# ============================================================
# CLEAN ID
# ============================================================

styles_df["id"] = pd.to_numeric(
    styles_df["id"],
    errors="coerce"
)

styles_df = styles_df.dropna(
    subset=["id"]
)

styles_df["id"] = styles_df[
    "id"
].astype(int)


# ============================================================
# MERGE
# ============================================================

print("Merging product metadata...")

df = pd.merge(
    styles_df,
    images_df,
    on="id",
    how="inner"
)


# ============================================================
# IMAGE PATH
# ============================================================

df["image_path"] = df[
    "filename"
].apply(
    lambda x: os.path.join(
        DATASET_DIR,
        "images",
        x
    )
)


# ============================================================
# READ PRICE FROM JSON
# ============================================================

prices = []
discounted_prices = []


for product_id in df["id"]:

    json_path = os.path.join(
        STYLES_DIR,
        f"{product_id}.json"
    )

    price = None
    discounted_price = None

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            product_json = json.load(f)

        product_data = product_json.get(
            "data",
            {}
        )

        price = product_data.get(
            "price"
        )

        discounted_price = product_data.get(
            "discountedPrice"
        )

    except Exception:

        pass

    prices.append(price)

    discounted_prices.append(
        discounted_price
    )


df["price"] = prices

df["discounted_price"] = discounted_prices


# ============================================================
# SELECT COLUMNS
# ============================================================

df = df[
    [
        "id",
        "image_path",
        "filename",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
        "price",
        "discounted_price",
        "link"
    ]
]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    PRODUCTS_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("DATA PREPARATION COMPLETE")
print("========================================")

print(
    f"Products: {len(df)}"
)

print(
    f"Output: {PRODUCTS_CSV}"
)

print()
print(
    df.head()
)