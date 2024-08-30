import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional product review writer. Generate detailed and realistic product reviews based on the provided product details."),
        ("user", "Product Name: {product_name}\nFeatures: {features}\nCustomer Sentiment: {sentiment}\nOverall Rating: {rating}\n\nPlease generate a comprehensive and balanced product review based on the above details.")
    ]
)

st.set_page_config(page_title='Generative AI Product Review Synthesizer', layout='wide')
st.title('Generative AI Product Review Synthesizer')
st.markdown("Welcome to the **Generative AI Product Review Synthesizer**. Enter product details below to generate a realistic review based on the provided features, sentiment, and rating.")

st.header("Enter Product Details")

col1, col2 = st.columns([2, 1])

with col1:
    product_name = st.text_input("Product Name", placeholder="e.g., SuperWidget 3000")
    features = st.text_area("Features (comma-separated)", placeholder="e.g., high durability, sleek design, user-friendly interface", height=150)

with col2:
    sentiment = st.selectbox("Sentiment", ["positive", "neutral", "negative"])
    rating = st.slider("Rating (1 to 5)", min_value=1, max_value=5)

llm = ChatGroq(model="llama3-8b-8192")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if product_name and features:
    with st.spinner('Generating your product review...'):
        review = chain.invoke({
            'product_name': product_name,
            'features': features,
            'sentiment': sentiment,
            'rating': rating
        })
    st.subheader("Generated Product Review:")
    st.write(review)
else:
    st.warning("Please fill in all the required fields to generate a product review.")

