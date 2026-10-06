import torch
from PIL import Image

from clip_encoder import (
    model,
    processor,
    DEVICE
)


CATEGORIES = [
    "Accessory Gift Set",
    "Baby Dolls",
    "Backpacks",
    "Bangle",
    "Basketballs",
    "Bath Robe",
    "Beauty Accessory",
    "Belts",
    "Blazers",
    "Body Lotion",
    "Body Wash and Scrub",
    "Booties",
    "Boxers",
    "Bra",
    "Bracelet",
    "Briefs",
    "Camisoles",
    "Capris",
    "Caps",
    "Casual Shoes",
    "Churidar",
    "Clothing Set",
    "Clutches",
    "Compact",
    "Concealer",
    "Cufflinks",
    "Cushion Covers",
    "Deodorant",
    "Dresses",
    "Duffel Bag",
    "Dupatta",
    "Earrings",
    "Eye Cream",
    "Eyeshadow",
    "Face Moisturisers",
    "Face Scrub and Exfoliator",
    "Face Serum and Gel",
    "Face Wash and Cleanser",
    "Flats",
    "Flip Flops",
    "Footballs",
    "Formal Shoes",
    "Foundation and Primer",
    "Fragrance Gift Set",
    "Free Gifts",
    "Gloves",
    "Hair Accessory",
    "Hair Colour",
    "Handbags",
    "Hat",
    "Headband",
    "Heels",
    "Highlighter and Blush",
    "Innerwear Vests",
    "Ipad",
    "Jackets",
    "Jeans",
    "Jeggings",
    "Jewellery Set",
    "Jumpsuit",
    "Kajal and Eyeliner",
    "Key chain",
    "Kurta Sets",
    "Kurtas",
    "Kurtis",
    "Laptop Bag",
    "Leggings",
    "Lehenga Choli",
    "Lip Care",
    "Lip Gloss",
    "Lip Liner",
    "Lip Plumper",
    "Lipstick",
    "Lounge Pants",
    "Lounge Shorts",
    "Lounge Tshirts",
    "Makeup Remover",
    "Mascara",
    "Mask and Peel",
    "Mens Grooming Kit",
    "Messenger Bag",
    "Mobile Pouch",
    "Mufflers",
    "Nail Essentials",
    "Nail Polish",
    "Necklace and Chains",
    "Nehru Jackets",
    "Night suits",
    "Nightdress",
    "Patiala",
    "Pendant",
    "Perfume and Body Mist",
    "Rain Jacket",
    "Rain Trousers",
    "Ring",
    "Robe",
    "Rompers",
    "Rucksacks",
    "Salwar",
    "Salwar and Dupatta",
    "Sandals",
    "Sarees",
    "Scarves",
    "Shapewear",
    "Shirts",
    "Shoe Accessories",
    "Shoe Laces",
    "Shorts",
    "Shrug",
    "Skirts",
    "Socks",
    "Sports Sandals",
    "Sports Shoes",
    "Stockings",
    "Stoles",
    "Suits",
    "Sunglasses",
    "Sunscreen",
    "Suspenders",
    "Sweaters",
    "Sweatshirts",
    "Swimwear",
    "Tablet Sleeve",
    "Ties",
    "Ties and Cufflinks",
    "Tights",
    "Toner",
    "Tops",
    "Track Pants",
    "Tracksuits",
    "Travel Accessory",
    "Trolley Bag",
    "Trousers",
    "Trunk",
    "Tshirts",
    "Tunics",
    "Umbrellas",
    "Waist Pouch",
    "Waistcoat",
    "Wallets",
    "Watches",
    "Water Bottle",
    "Wristbands"
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