import streamlit as st
import os
import sys
import hashlib

# Allow imports from src/
sys.path.append("src")

from PIL import Image

from classifier import predict_category
from color import predict_color
from search import retrieve_products
from ranking import rerank_products


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fashion Finder",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("👗 Fashion Finder")

st.write(
    "Upload a fashion image and find similar products "
    "using CLIP, FAISS and hybrid ranking."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # Create a unique ID for the uploaded image
    image_bytes = uploaded_file.getvalue()

    image_id = hashlib.md5(
        image_bytes
    ).hexdigest()


    # ========================================================
    # PROCESS IMAGE ONLY WHEN A NEW IMAGE IS UPLOADED
    # ========================================================

    if (
        "image_id" not in st.session_state
        or st.session_state.image_id != image_id
    ):

        # Save uploaded image temporarily
        temp_path = "temp_query.jpg"

        with open(temp_path, "wb") as f:
            f.write(image_bytes)


        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            temp_path
        ).convert("RGB")


        # ----------------------------------------------------
        # Predict category
        # ----------------------------------------------------

        with st.spinner(
            "Detecting product category..."
        ):

            query_category, category_confidence = (
                predict_category(
                    image
                )
            )


        # ----------------------------------------------------
        # Detect color
        # ----------------------------------------------------

        with st.spinner(
            "Detecting product color..."
        ):

            query_color, color_confidence = (
                predict_color(
                    image
                )
            )


        # ----------------------------------------------------
        # Retrieve candidates using FAISS
        # ----------------------------------------------------

        with st.spinner(
            "Finding visually similar products..."
        ):

            candidates = retrieve_products(
                temp_path,
                k=200
            )


        # ----------------------------------------------------
        # Store everything in session state
        # ----------------------------------------------------

        st.session_state.image_id = image_id

        st.session_state.query_category = (
            query_category
        )

        st.session_state.category_confidence = (
            category_confidence
        )

        st.session_state.query_color = (
            query_color
        )

        st.session_state.color_confidence = (
            color_confidence
        )

        st.session_state.candidates = (
            candidates
        )


        # Remove temporary file
        if os.path.exists(temp_path):

            os.remove(temp_path)


    # ========================================================
    # WEIGHT CONTROLS
    # ========================================================

    st.divider()

    st.subheader("⚙️ Ranking Weights")

    st.write(
        "Adjust the importance of each factor. "
        "The weights are automatically normalized to 100%."
    )


    weight_col1, weight_col2, weight_col3, weight_col4 = (
        st.columns(4)
    )


    with weight_col1:

        visual_weight = st.slider(
            "Visual Similarity",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )


    with weight_col2:

        category_weight = st.slider(
            "Category",
            min_value=0,
            max_value=100,
            value=35,
            step=5
        )


    with weight_col3:

        color_weight = st.slider(
            "Color",
            min_value=0,
            max_value=100,
            value=10,
            step=5
        )


    with weight_col4:

        price_weight = st.slider(
            "Price",
            min_value=0,
            max_value=100,
            value=5,
            step=5
        )


    # ========================================================
    # NORMALIZE WEIGHTS
    # ========================================================

    total_weight = (
        visual_weight
        + category_weight
        + color_weight
        + price_weight
    )


    if total_weight == 0:

        st.warning(
            "Please give at least one ranking factor "
            "a non-zero weight."
        )

        st.stop()


    normalized_visual = (
        visual_weight / total_weight
    )

    normalized_category = (
        category_weight / total_weight
    )

    normalized_color = (
        color_weight / total_weight
    )

    normalized_price = (
        price_weight / total_weight
    )


    # Show effective weights
    st.caption(
        f"Effective weights → "
        f"Visual: {normalized_visual:.0%} | "
        f"Category: {normalized_category:.0%} | "
        f"Color: {normalized_color:.0%} | "
        f"Price: {normalized_price:.0%}"
    )


    # ========================================================
    # QUERY INFORMATION
    # ========================================================

    st.divider()

    col1, col2 = st.columns([1, 2])


    with col1:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )


    with col2:

        st.subheader("🔍 Detected Product")


        st.write(
            f"**Category:** "
            f"{st.session_state.query_category}"
        )


        st.write(
            f"**Category Confidence:** "
            f"{st.session_state.category_confidence:.2f}"
        )


        st.write(
            f"**Color:** "
            f"{st.session_state.query_color}"
        )


        st.write(
            f"**Color Confidence:** "
            f"{st.session_state.color_confidence:.2f}"
        )


    # ========================================================
    # RERANK PRODUCTS
    # ========================================================

    ranked_products = rerank_products(
        st.session_state.candidates,

        query_category=(
            st.session_state.query_category
        ),

        query_color=(
            st.session_state.query_color
        ),

        max_price=None,

        visual_weight=normalized_visual,

        category_weight=normalized_category,

        color_weight=normalized_color,

        price_weight=normalized_price
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    st.subheader("✨ Recommended Products")


    products = ranked_products[:10]


    # Create 5 columns
    cols = st.columns(5)


    for i, product in enumerate(products):

        with cols[i % 5]:

            image_path = product["image_path"]


            # ------------------------------------------------
            # Product image
            # ------------------------------------------------

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    use_container_width=True
                )


            # ------------------------------------------------
            # Product name
            # ------------------------------------------------

            st.markdown(
                f"**{product['product_name']}**"
            )


            # ------------------------------------------------
            # Product information
            # ------------------------------------------------

            st.write(
                f"Category: "
                f"{product['category']}"
            )


            st.write(
                f"Color: "
                f"{product['color']}"
            )


            st.write(
                f"Price: "
                f"₹{product['price']}"
            )


            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            st.caption(
                f"Visual: "
                f"{product['visual_score']:.3f}"
            )


            st.caption(
                f"Category: "
                f"{product['category_score']:.3f}"
            )


            st.caption(
                f"Color: "
                f"{product['color_score']:.3f}"
            )


            st.caption(
                f"Final Score: "
                f"{product['final_score']:.3f}"
            )


            st.divider()