# Customer Support Ticket Classifier

A customer support ticket classification project created for **PROG74040 – Advanced Topics in Artificial Intelligence and Machine Learning**.

The application uses a fine-tuned **DistilBERT** model to classify support tickets into four categories:

- Change
- Incident
- Problem
- Request

## Live App

https://customer-support-ticket-classifier-g6cwpvwwbfyr5fyw2vmv6m.streamlit.app/

## How It Works

Users enter a ticket subject and description. The text is processed by the trained DistilBERT model, which returns:

- Predicted ticket type
- Confidence score

## Deployment

- **GitHub** – application code
- **Hugging Face** – trained DistilBERT model
- **Streamlit Community Cloud** – deployed web app

Model repository:

https://huggingface.co/ToastyKrawch/customer-support-ticket-distilbert

## Files

```
app.py
requirements.txt
README.md
```

## Team
Group 8
- Bradley Krawchyk
- Ekam Lally
- Nick Packull-McCormick
