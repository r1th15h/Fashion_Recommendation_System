import os
import csv


DATASET_DIR = "fashion-dataset"

IMAGES_DIR = os.path.join(
    DATASET_DIR,
    "images"
)

STYLES_DIR = os.path.join(
    DATASET_DIR,
    "styles"
)

IMAGES_CSV = os.path.join(
    DATASET_DIR,
    "images.csv"
)

STYLES_CSV = os.path.join(
    DATASET_DIR,
    "styles.csv"
)


def verify_dataset():

    print()
    print("========================================")
    print("DATASET VERIFICATION")
    print("========================================")
    print()


    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    print("Checking directories...")

    print(
        "Images directory:",
        os.path.exists(IMAGES_DIR)
    )

    print(
        "Styles directory:",
        os.path.exists(STYLES_DIR)
    )


    # --------------------------------------------------------
    # Check CSV files
    # --------------------------------------------------------

    print()
    print("Checking CSV files...")

    print(
        "images.csv:",
        os.path.exists(IMAGES_CSV)
    )

    print(
        "styles.csv:",
        os.path.exists(STYLES_CSV)
    )


    # --------------------------------------------------------
    # Count images
    # --------------------------------------------------------

    print()
    print("Counting files...")

    image_files = [
        filename
        for filename in os.listdir(IMAGES_DIR)
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    json_files = [
        filename
        for filename in os.listdir(STYLES_DIR)
        if filename.lower().endswith(".json")
    ]


    print(
        "Image files:",
        len(image_files)
    )

    print(
        "Style JSON files:",
        len(json_files)
    )


    # --------------------------------------------------------
    # Read image IDs
    # --------------------------------------------------------

    print()
    print("Reading images.csv...")

    image_ids = set()

    with open(
        IMAGES_CSV,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(f)

        print(
            "CSV columns:",
            reader.fieldnames
        )

        for row in reader:

            filename = row.get(
                "filename"
            )

            if not filename:
                continue

            try:

                product_id = int(
                    os.path.splitext(
                        filename
                    )[0]
                )

                image_ids.add(
                    product_id
                )

            except ValueError:

                pass


    # --------------------------------------------------------
    # Read style IDs
    # --------------------------------------------------------

    print()
    print("Reading styles.csv...")

    style_ids = set()

    with open(
        STYLES_CSV,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        reader = csv.DictReader(f)

        print(
            "CSV columns:",
            reader.fieldnames
        )

        for row in reader:

            product_id = row.get(
                "id"
            )

            if not product_id:
                continue

            try:

                style_ids.add(
                    int(product_id)
                )

            except ValueError:

                pass


    # --------------------------------------------------------
    # Check physical files
    # --------------------------------------------------------

    print()
    print("Checking matching files...")


    missing_images = []

    for product_id in image_ids:

        image_path = os.path.join(
            IMAGES_DIR,
            f"{product_id}.jpg"
        )

        if not os.path.exists(
            image_path
        ):

            missing_images.append(
                product_id
            )


    missing_styles = []

    for product_id in style_ids:

        json_path = os.path.join(
            STYLES_DIR,
            f"{product_id}.json"
        )

        if not os.path.exists(
            json_path
        ):

            missing_styles.append(
                product_id
            )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("========================================")
    print("RESULT")
    print("========================================")

    print(
        "IDs in images.csv:",
        len(image_ids)
    )

    print(
        "IDs in styles.csv:",
        len(style_ids)
    )

    print(
        "Missing images:",
        len(missing_images)
    )

    print(
        "Missing style JSON files:",
        len(missing_styles)
    )


    # --------------------------------------------------------
    # Show examples of missing files
    # --------------------------------------------------------

    if missing_images:

        print()
        print(
            "First missing image IDs:"
        )

        print(
            missing_images[:20]
        )


    if missing_styles:

        print()
        print(
            "First missing style IDs:"
        )

        print(
            missing_styles[:20]
        )


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()

    if (
        len(missing_images) == 0
        and len(missing_styles) == 0
    ):

        print(
            "Dataset verification passed!"
        )

    else:

        print(
            "Dataset has missing files."
        )


if __name__ == "__main__":

    verify_dataset()