"""
Plant Disease Recognition System (Streamlit app)

Run, from this folder with the virtual environment active:
    python -m streamlit run main.py
"""
from pathlib import Path

import streamlit as st
from PIL import Image

from plant_labels import ModelProblem, is_healthy, load_model, predict, pretty

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Plant Disease Recognition", page_icon="🌿")


@st.cache_resource(show_spinner="Loading the model (first time only)...")
def get_model():
    return load_model(HERE)


# Sidebar
st.sidebar.title("Dashboard") 
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    banner = HERE / "home_page.jpeg"
    if banner.exists():
        st.image(str(banner))
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant leaf, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant leaf with suspected disease.
    2. **Analysis:** Our system will process the image using a convolutional neural network (CNN) to identify potential diseases.
    3. **Results:** View the predicted disease, how sure the model is, and the other likely possibilities.

    ### Why Choose Us?
    - **Accuracy:** A CNN trained on about 87,000 leaf images (around 94% validation accuracy after 10 epochs of training).
    - **User-Friendly:** Simple and intuitive interface.
    - **Fast and Efficient:** Results in seconds.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image.

    ### About
    Learn more about the dataset and the model on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    The *New Plant Diseases Dataset* (Kaggle), recreated using offline augmentation from the
    original PlantVillage dataset. It has about 87K RGB images of healthy and diseased crop
    leaves in **38 classes** (14 plants: apple, blueberry, cherry, corn, grape, orange, peach,
    pepper, potato, raspberry, soybean, squash, strawberry, tomato).

    #### Content
    1. train (70,295 images)
    2. validation (17,572 images)
    3. test (33 images)

    #### Model
    A CNN with five convolution blocks (32 → 512 filters), dropout, one dense layer of 1,500
    neurons and a 38-way softmax output. Input images are 128 × 128 pixels.

    #### Limits
    - The model gives **one answer per image**, so use a photo of a single leaf.
    - It was trained on clear, close-up photos; messy backgrounds reduce accuracy.
    - Only the 14 plants above are known (there is no banana, for example).
    - Educational tool, not a substitute for an agriculture expert.
    """)

else:  # Disease Recognition
    st.header("Disease Recognition")
    try:
        model = get_model()
    except ModelProblem as e:
        st.error(str(e))
        st.stop()

    uploaded = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png", "bmp", "webp"])
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            image.load()
        except Exception:  # noqa: BLE001
            st.error("That file could not be read as an image. Try a JPG or PNG.")
            image = None
    if image is not None:
                st.image(image.convert("RGB"), caption="Your image", output_format="PNG")

    if st.button("Predict", type="primary", disabled=image is None):
        with st.spinner("Analysing..."):
            results = predict(model, image, top=3)
        best, prob = results[0]
        plant, disease = pretty(best)
        if is_healthy(best):
            st.success(f"Our prediction: **{plant}**: healthy leaf  ({prob * 100:.1f}% sure)")
        else:
            st.warning(f"Our prediction: **{plant}**: {disease}  ({prob * 100:.1f}% sure)")
        if prob < 0.5:
            st.info("The model is not very sure about this one. Try a clearer photo of a single leaf.")
        st.write("Top 3 possibilities")
        st.table([{"Plant": pretty(n)[0], "Condition": pretty(n)[1] or "healthy",
                   "Confidence": f"{p * 100:.1f}%"} for n, p in results])
    elif image is None:
        st.caption("Upload an image, then press Predict.")
