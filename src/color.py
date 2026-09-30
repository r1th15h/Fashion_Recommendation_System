import torch
from PIL import Image

from clip_encoder import (
    model,
    processor,
    DEVICE
)


COLORS = [
    "black",
    "white",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "maroon"
]


def predict_color(image: Image.Image):

    prompts = [
        f"a photo of a {color} clothing item"
        for color in COLORS
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
        COLORS[best_index].capitalize(),
        confidence
    )