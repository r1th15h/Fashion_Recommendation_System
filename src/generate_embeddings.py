import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from clip_encoder import processor, model, DEVICE

from config import (
    PRODUCTS_CSV,
    PRODUCT_EMBEDDINGS,
    PRODUCT_IDS
)


BATCH_SIZE = 16

CHECKPOINT_EVERY = 100


def save_checkpoint(
    embeddings,
    product_ids
):

    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    np.save(
        PRODUCT_EMBEDDINGS,
        np.array(
            embeddings,
            dtype="float32"
        )
    )

    np.save(
        PRODUCT_IDS,
        np.array(
            product_ids,
            dtype="int64"
        )
    )


def generate_embeddings():

    df = pd.read_csv(
        PRODUCTS_CSV
    )

    embeddings = []
    product_ids = []

    total = len(df)

    print(
        f"Total products: {total}"
    )

    for start in tqdm(
        range(0, total, BATCH_SIZE),
        desc="Generating embeddings"
    ):

        batch_df = df.iloc[
            start:start + BATCH_SIZE
        ]

        images = []
        valid_ids = []

        for _, row in batch_df.iterrows():

            image_path = row[
                "image_path"
            ]

            if not os.path.exists(
                image_path
            ):

                print(
                    f"Skipping missing image: "
                    f"{image_path}"
                )

                continue

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                images.append(
                    image
                )

                valid_ids.append(
                    int(row["id"])
                )

            except Exception as e:

                print(
                    f"Skipping {image_path}: {e}"
                )


        if not images:
            continue


        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        pixel_values = (
            inputs["pixel_values"]
            .to(DEVICE)
        )


        with torch.no_grad():

            vision_outputs = (
                model.vision_model(
                    pixel_values=pixel_values
                )
            )

            pooled_output = (
                vision_outputs.pooler_output
            )

            batch_embeddings = (
                model.visual_projection(
                    pooled_output
                )
            )


        batch_embeddings = (
            batch_embeddings
            /
            batch_embeddings.norm(
                p=2,
                dim=-1,
                keepdim=True
            )
        )


        embeddings.extend(
            batch_embeddings
            .cpu()
            .numpy()
        )

        product_ids.extend(
            valid_ids
        )


        # Checkpoint periodically
        batch_number = (
            start // BATCH_SIZE
        ) + 1

        if (
            batch_number
            % CHECKPOINT_EVERY
            == 0
        ):

            save_checkpoint(
                embeddings,
                product_ids
            )

            print(
                f"\nCheckpoint saved: "
                f"{len(embeddings)} embeddings"
            )


    # Final save

    save_checkpoint(
        embeddings,
        product_ids
    )


    print()
    print(
        "========================================"
    )

    print(
        "EMBEDDING GENERATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        "Embeddings shape:",
        np.array(
            embeddings
        ).shape
    )

    print(
        "Product IDs shape:",
        np.array(
            product_ids
        ).shape
    )

    print(
        "Saved embeddings:",
        PRODUCT_EMBEDDINGS
    )

    print(
        "Saved IDs:",
        PRODUCT_IDS
    )


if __name__ == "__main__":

    generate_embeddings()