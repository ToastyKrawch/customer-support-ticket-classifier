import streamlit as st
import torch
import qrcode
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_REPO = "ToastyKrawch/customer-support-ticket-distilbert"

APP_URL = (
    "https://customer-support-ticket-classifier-"
    "g6cwpvwwbfyr5fyw2vmv6m.streamlit.app/"
)

MAX_LENGTH = 256


st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    layout="centered"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_REPO
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

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


# --------------------------------------------------
# QR code
# --------------------------------------------------

@st.cache_data
def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=None,
        box_size=8,
        border=4
    )

    qr.add_data(url)
    qr.make(fit=True)

    qr_image = qr.make_image(
        fill_color="black",
        back_color="white"
    )

    buffer = BytesIO()
    qr_image.save(buffer, format="PNG")

    return buffer.getvalue()


qr_code = generate_qr_code(APP_URL)


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("Customer Support Ticket Classifier")

intro_col, qr_col = st.columns([2.3, 1])

with intro_col:
    st.write(
        "Enter the subject and description of a customer support ticket. "
        "The model will classify the ticket into one of four categories."
    )

    st.write(
        "Our classifier uses a fine-tuned **DistilBERT** model to analyze "
        "the customer's written ticket and predict its most likely type."
    )

with qr_col:
    st.markdown("#### Try it yourself")

    st.image(
        qr_code,
        width=175
    )

    st.link_button(
        "Open App",
        APP_URL,
        use_container_width=True
    )


# --------------------------------------------------
# Ticket type information
# --------------------------------------------------

with st.expander("What do the ticket types mean?"):
    st.markdown(
        """
        - **Incident** — An unexpected issue or service interruption that needs attention.
        - **Request** — A user asking for access, information, software, or another service.
        - **Change** — A request to modify an existing system, configuration, or service.
        - **Problem** — A recurring or underlying issue that may require further investigation.
        """
    )


st.divider()


# --------------------------------------------------
# Ticket input form
# --------------------------------------------------

with st.form("ticket_form"):

    subject = st.text_input(
        "Ticket Subject",
        placeholder="Example: Unable to connect to VPN"
    )

    body = st.text_area(
        "Ticket Description",
        height=180,
        placeholder="Describe the issue or request here..."
    )

    submitted = st.form_submit_button(
        "Classify Ticket",
        use_container_width=True
    )


# --------------------------------------------------
# Prediction output
# --------------------------------------------------

if submitted:

    if not subject.strip() and not body.strip():

        st.warning(
            "Please enter a ticket subject or description."
        )

    else:

        with st.spinner("Analyzing ticket..."):

            predicted_label, confidence = predict_ticket(
                subject,
                body
            )

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Ticket Type",
                predicted_label
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        st.progress(
            confidence,
            text=f"Model confidence: {confidence:.2%}"
        )

        st.caption(
            "The prediction is generated by a fine-tuned DistilBERT "
            "model and should be treated as decision support rather "
            "than a guaranteed classification."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "PROG74040 — Advanced Topics in Artificial Intelligence "
    "and Machine Learning | Group 8"
)
