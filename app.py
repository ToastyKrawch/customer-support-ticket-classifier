import streamlit as st

st.set_page_config(
    page_title="Customer Support Ticket Classifier",
    page_icon="🎫"
)

st.title("Customer Support Ticket Classifier")

st.write(
    "Enter a customer support ticket below to classify it as "
    "Change, Incident, Problem, or Request."
)

subject = st.text_input("Ticket Subject")

body = st.text_area(
    "Ticket Description",
    height=200
)

if st.button("Classify Ticket"):
    if not subject.strip() and not body.strip():
        st.warning("Please enter a ticket subject or description.")
    else:
        st.info("Model prediction will be added in the next step.")
