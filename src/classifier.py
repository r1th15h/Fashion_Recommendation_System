import torch
from PIL import Image

from clip_encoder import (
    model,
    processor,
    DEVICE
)


CATEGORIES = [
    "shirt",
    "t-shirt",
    "dress",
    "skirt",
    "jeans",
    "trousers",
    "pants",
    "jacket",
    "coat",
    "sweater",
    "hoodie",
    "top",
    "shorts",
    "saree",
    "kurta",
    "blazer",
    "shoes",
    "sandals",
    "watch",
    "bag"
]


def predict_category(
    image: Image.Image
):

    prompts = [
        f"a photo of a {category}"
        for category in CATEGORIES
    ]

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probabilities = (
            outputs.logits_per_image
            .softmax(dim=1)
        )

    best_index = probabilities.argmax(
        dim=1
    ).item()

    confidence = probabilities[
        0,
        best_index
    ].item()

    return (
        CATEGORIES[best_index],
        confidence
    )