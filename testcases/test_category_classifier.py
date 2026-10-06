import os
import sys

import pandas as pd

from PIL import Image


# ============================================================
# ALLOW IMPORTS FROM src/
# ============================================================

sys.path.append("src")

from classifier import predict_category


# ============================================================
# TEST CASE FILE
# ============================================================

TEST_CASES = (
    "testcases/category_test_cases.csv"
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    df = pd.read_csv(
        TEST_CASES
    )

    total = 0
    correct = 0

    print()
    print("=" * 70)
    print("CATEGORY CLASSIFIER TEST")
    print("=" * 70)

    for _, row in df.iterrows():

        image_path = os.path.join(
            "testcases",
            row["image"]
        )

        expected = str(
            row["expected_category"]
        ).strip().lower()


        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not os.path.exists(
            image_path
        ):

            print(
                f"[SKIP] Missing image: "
                f"{image_path}"
            )

            continue


        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # PREDICT CATEGORY
        # ----------------------------------------------------

        predicted, confidence = (
            predict_category(
                image
            )
        )


        predicted = str(
            predicted
        ).strip().lower()


        # ----------------------------------------------------
        # CHECK RESULT
        # ----------------------------------------------------

        passed = (
            predicted == expected
        )

        total += 1

        if passed:
            correct += 1


        status = (
            "PASS"
            if passed
            else "FAIL"
        )


        print(
            f"[{status}] "
            f"{row['image']} | "
            f"Expected: {expected} | "
            f"Predicted: {predicted} | "
            f"Confidence: {confidence:.3f}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 70)

    print(
        f"Correct: {correct}/{total}"
    )

    if total > 0:

        accuracy = (
            correct / total
        )

        print(
            f"Category Accuracy: "
            f"{accuracy:.2%}"
        )

    else:

        print(
            "Category Accuracy: N/A"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()