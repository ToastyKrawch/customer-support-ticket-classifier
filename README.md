# Customer Support Ticket Classifier

A deployed natural language processing (NLP) application that classifies customer support tickets as **Change**, **Incident**, **Problem**, or **Request**. Users enter a subject and description, and the app returns a predicted ticket type with a confidence score.

**[Try the live Streamlit app](https://customer-support-ticket-classifier-g6cwpvwwbfyr5fyw2vmv6m.streamlit.app/)** · **[View the trained model on Hugging Face](https://huggingface.co/ToastyKrawch/customer-support-ticket-distilbert)**

## Project highlights

- Built a browser-based support ticket classification interface with Streamlit.
- Integrated a fine-tuned DistilBERT model hosted on Hugging Face using PyTorch and Transformers.
- Displayed ticket classifications and confidence scores, with a QR code for mobile access during demonstrations.
- Deployed the application through GitHub and Streamlit Community Cloud.

## How it works

1. Enter a customer support ticket's subject and description.
2. The app combines the two fields and tokenizes the text, truncating or padding to 256 tokens.
3. A fine-tuned DistilBERT sequence-classification model predicts one of four ticket types.
4. The interface displays the predicted category and its model confidence.

The app is intended as a decision-support demonstration; predictions are not guaranteed to be correct.

## Model comparison

As part of the wider team project, a TF-IDF + Logistic Regression baseline was compared with fine-tuned DistilBERT. The team's Phase 2 report recorded these held-out test results:

| Metric | TF-IDF + Logistic Regression | Fine-tuned DistilBERT |
| --- | ---: | ---: |
| Accuracy | 86.41% | 88.80% |
| Weighted F1 | 85.60% | 88.83% |
| Macro F1 | 86.16% | 89.73% |

These values are taken from the team's course report and are not re-computed by this application. The training and evaluation notebooks are not included in this deployment repository.

## Tech stack

- **Application:** Python, Streamlit
- **Model inference:** PyTorch, Hugging Face Transformers, DistilBERT
- **Deployment:** GitHub, Hugging Face Model Hub, Streamlit Community Cloud
- **Interface:** Streamlit form, predicted-label and confidence displays, QR code

## Run locally

Use a Python environment with dependencies compatible with `requirements.txt`.

```bash
git clone https://github.com/ToastyKrawch/customer-support-ticket-classifier.git
cd customer-support-ticket-classifier
python -m venv .venv
```

Activate the environment (on Windows: `.venv\Scripts\activate`; on macOS/Linux: `source .venv/bin/activate`) and then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first startup, the application downloads the tokenizer and trained model from Hugging Face; an internet connection is required. Model loading and inference can take time depending on the environment.

## Repository files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, model loading, inference, confidence display, QR code |
| `requirements.txt` | Python application dependencies |
| `README.md` | Project overview and setup instructions |

## Team and contributions

Created for **PROG74040 – Advanced Topics in Artificial Intelligence and Machine Learning** at Conestoga College (Group 8).

- **Bradley Krawchyk:** System architecture and application deployment; Streamlit interface, integration with the Hugging Face-hosted DistilBERT model, Streamlit Community Cloud deployment, deployed-app testing, and QR code/demo support.
- **Ekam Lally:** Baseline and advanced model development, fine-tuning, comparison, and evaluation.
- **Nick Packull-McCormick:** Dataset sourcing, cleaning, preparation, and initial DistilBERT testing.

This repository primarily contains the deployed application, not the full training pipeline.
