import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

model_image_segmentation = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
model_image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
model_summarization = pipeline("summarization", model="facebook/bart-large-cnn")

st.title("Segmentation d'images et résumé")

with st.form('form'):
    uploaded_file = st.file_uploader("Choisir une image")
    submit = st.form_submit_button("Analyser l'image")

if submit:
    if not uploaded_file:
        st.error("Vous devez choisir une image")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.write("Segmentation de l'image en cours")

        segments = model_image_segmentation(image)

        descriptions = []
        for segment in segments:
            score = segment['score']
            label = segment['label']
            mask = segment['mask']

            mask_np = np.array(mask)
            ys, xs = np.where(mask_np > 0)

            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            crop = image.crop((xmin, ymin, xmax, ymax))

            image_to_text_response = model_image_to_text(crop)
            description = None
            if image_to_text_response:
                description = image_to_text_response[0]['generated_text']

            st.image(crop)
            st.write(f"Label : {label}")
            if description:
                st.write(f"Description : {description}")
                descriptions.append(description)
            else:
                st.write(f"Aucune description")

        if descriptions:
            st.subheader("Résumé de l'image")
            st.write(f"En attente ...")

            text = " ".join(descriptions)

            input_length = len(text.split())
            max_length = min(60, max(10, input_length))
            summary_response = model_summarization(text, max_length=max_length, min_length=10)
            summary = None
            if summary_response:
                summary = summary_response[0]['summary_text']

            if summary:
                st.write(summary)
            else:
                st.error("Pas de résumé")
