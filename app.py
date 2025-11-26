import numpy as np
import streamlit as st
from PIL import Image
from transformers import pipeline

st.title("Segmentation d'images et résumé")

with st.form('form'):
    uploaded_file = st.file_uploader("Choisir une image")
    submit = st.form_submit_button("Analyser l'image")

if submit:
    if not uploaded_file:
        st.error("Vous devez choisir une image")
    else:
        try:
            model_image_segmentation = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
        except Exception as e:
            st.error(f"Impossible de charger le model image-segmentation : {e}")
            st.stop()

        try:
            model_image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        except Exception as e:
            st.error(f"Impossible de charger le model image-to-text : {e}")
            st.stop()

        try:
            model_summarization = pipeline("summarization", model="facebook/bart-large-cnn")
        except Exception as e:
            st.error(f"Impossible de charger le model summarization : {e}")
            st.stop()

        st_segmentation_pending = st.empty()
        st_segmentation_pending.write("Analyse de l'image en cours")

        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Impossible de lire l'image : {e}")
            st.stop()

        try:
            segments = model_image_segmentation(image)
        except Exception as e:
            st.error(f"Erreur de segmentation :{e}")
            st.stop()

        descriptions = []
        for segment in segments:
            score = segment['score']
            label = segment['label']
            mask = segment['mask']

            # CROP IMAGE A PARTIR DU MASK
            mask_np = np.array(mask)
            ys, xs = np.where(mask_np > 0)
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            crop = image.crop((xmin, ymin, xmax, ymax))

            description = None
            try:
                image_to_text_response = model_image_to_text(crop)
                if image_to_text_response:
                    description = image_to_text_response[0]['generated_text']
            except Exception as e:
                pass

            st.image(crop)
            st.write(f"Label : {label}")
            if description:
                st.write(f"Description : {description}")
                descriptions.append(description)
            else:
                st.error(f"Aucune description")

        st_segmentation_pending.empty()

        if descriptions:
            st.subheader("Résumé de l'image")
            st.image(uploaded_file)

            st_pending = st.empty()
            st_pending.write("En attente ...")

            text = " ".join(descriptions)

            input_length = len(text.split())
            max_length = 100
            if input_length < max_length:
                max_length = input_length

            summary = None

            try:
                summary_response = model_summarization(text, max_length=max_length, min_length=10)
                if summary_response:
                    summary = summary_response[0]['summary_text']
            except Exception as e:
                st.error(f"Erreur résumé :{e}")

            st_pending.empty()
            if summary:
                st.write(summary)
            else:
                st.error("Pas de résumé")
