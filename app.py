import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_REPO = "ToastyKrawch/customer-support-ticket-distilbert"
MAX_LENGTH = 256


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


def predict_ticket(subject, body):
    # Match the preprocessing used during training
    text = f"{subject.strip()} {body.strip()}".strip()

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

        predicted_id = torch.argmax(
            probabilities
        ).item()

    predicted_label = model.config.id2label[predicted_id]

    confidence = probabilities[predicted_id].item()

    return predicted_label, confidence


st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫"
)

st.title("Customer Support Ticket Classifier")

st.write(
    "Enter a customer support ticket below to classify it as "
    "Change, Incident, Problem, or Request."
)

subject = st.text_input(
    "Ticket Subject"
)

body = st.text_area(
    "Ticket Description",
    height=200
)

if st.button("Classify Ticket"):

    if not subject.strip() and not body.strip():
        st.warning(
            "Please enter a ticket subject or description."
        )

    else:
        predicted_label, confidence = predict_ticket(
            subject,
            body
        )

        st.success(
            f"Predicted Ticket Type: {predicted_label}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )
