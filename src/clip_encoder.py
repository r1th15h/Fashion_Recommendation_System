import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"


processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model = CLIPModel.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()


def get_image_embedding(image: Image.Image):

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

        pooled_output = vision_outputs.pooler_output

        embedding = model.visual_projection(
            pooled_output
        )

    embedding = embedding / embedding.norm(
        p=2,
        dim=-1,
        keepdim=True
    )

    return embedding.cpu().numpy()