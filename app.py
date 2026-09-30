import streamlit as st
import os
import sys

sys.path.append("src")

from pipeline import search_products


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
    "from the Myntra fashion dataset."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # Temporary file for the retrieval pipeline
    temp_path = "temp_query.jpg"

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # ========================================================
    # DISPLAY QUERY IMAGE
    # ========================================================

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )


    # ========================================================
    # RUN ML PIPELINE
    # ========================================================

    with st.spinner(
        "Analyzing image and finding similar products..."
    ):

        result = search_products(
            temp_path
        )


    # ========================================================
    # QUERY INFORMATION
    # ========================================================

    with col2:

        st.subheader("🔍 Detected Product")

        query = result["query"]

        st.write(
            f"**Category:** {query['category']}"
        )

        st.write(
            f"**Category Confidence:** "
            f"{query['category_confidence']:.2f}"
        )

        st.write(
            f"**Color:** {query['color']}"
        )

        if "color_confidence" in query:

            st.write(
                f"**Color Confidence:** "
                f"{query['color_confidence']:.2f}"
            )


    st.divider()


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader("✨ Recommended Products")

    products = result["results"]


    # Create 5 product columns
    cols = st.columns(5)


    for i, product in enumerate(products):

        with cols[i % 5]:

            image_path = product["image_path"]


            # Product image
            if os.path.exists(image_path):

                st.image(
                    image_path,
                    use_container_width=True
                )


            # Product name
            st.markdown(
                f"**{product['product_name']}**"
            )


            # Product information
            st.write(
                f"Category: {product['category']}"
            )

            st.write(
                f"Color: {product['color']}"
            )

            st.write(
                f"Price: ₹{product['price']}"
            )


            # Similarity score
            st.caption(
                f"Visual similarity: "
                f"{product['visual_score']:.3f}"
            )


            st.divider()


    # ========================================================
    # REMOVE TEMPORARY FILE
    # ========================================================

    if os.path.exists(temp_path):

        os.remove(temp_path)