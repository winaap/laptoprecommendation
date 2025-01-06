import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

GOOGLE_API_KEY ="AIzaSyAmzB2ZO87tNY7ZTR8y30CTn2uuCIsfruo"
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(
    page_title="The Science of Choosing the Perfect Laptop",
    page_icon="💻",
    layout="wide"
)

# Load dataset
@st.cache_resource
def load_data():
    return pd.read_csv("laptops_clustered_with_labels.csv")

data = load_data()


# Dashboard title
st.title("The Science of Choosing the Perfect Laptop")
st.markdown("Explore the laptop market with insightful visualizations and get personalized recommendations tailored to your needs.")

# Query Section
st.markdown("### What Kind of Laptop Do You Need?")
user_query = st.text_input("Enter your requirements (e.g., 'gaming laptop under Rp10M with i5 processor'): ")


def get_gemini_recommendations(query, dataset):
    """Use Gemini AI to process the query with added context."""
    context = (
        "You are an AI assistant tasked with recommending laptops based on the given dataset. "
        "Please ensure all recommendations strictly follow the criteria provided in the query. "
        "For example, if the query specifies 'under 10M', only include laptops with prices below 10 million. "
        "Also, make sure to remove duplicate entries from the dataset before providing recommendations. "
        "Here are the first 5 rows of the dataset to help you understand its structure:\n"
    )
    context += dataset.head(5).to_csv(index=False)
    context += (
        "\nPlease recommend up to 3 laptops from the dataset based on the user's query. "
        "Include the following information for each recommended laptop: "
        "name, processor, RAM size, storage size and type, operating system, display size, price, and rating. "
        "Ensure all recommendations meet the constraints mentioned in the query. "
        "If there are no laptops that meet the criteria, explicitly state 'No laptops match the criteria.' "
        "Answer Example: Use this template to answer the query"
        "\n1. ASUS ROG Strix G15 (2022) Ryzen 7 Octa Core AMD R7 - 16GB RAM, 512GB SSD, Windows 11, 15.6-inch Display - Rp18,889,110 - Rating: 4.4"
        "\n2. ASUS TUF Gaming F15 Core i5 10th Gen - 8GB RAM, 512GB SSD, Windows 11, 15.6-inch Display - Rp9,443,610 - Rating: 4.4"
        "\n3. Lenovo V15 G2 Core i3 11th Gen - 8GB RAM, 1TB HDD + 256GB SSD, Windows 11, 15.6-inch Display - Rp7,084,125 - Rating: 4.4"
        "\n\nIMPORTANT NOTES:"
        "\n1. If there are duplicate laptops (rows with the same name, processor, price, and rating), only include one of them in the recommendations."
        "\n2. Ensure that all prices are displayed in a user-friendly format (e.g., Rp10,000,000). Avoid scientific notation or unreadable formats."
        "\n3. Provide each recommended laptop on a new line in the response. Do not list multiple laptops on the same line to avoid user confusion."
        "\n4. If the query explicitly states a price limit, processor type, or any other constraint, ensure that all recommendations satisfy these conditions strictly."
        "\n5. If there are less than 3 laptops that match the criteria, provide only the available matches."
    )

    
    api_key = "AIzaSyAQU5WKyxRoLw8AZ0JhhcVny09xSjzRN-0"
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    # Buat agen Pandas
    agent = create_pandas_dataframe_agent(
        llm,
        dataset,
        verbose=True,
        allow_dangerous_code=True,
        agent_type="zero-shot-react-description",
        handle_parsing_errors=True
    )
    try:
        response = agent.invoke(f"{context}\n\n{query}")
        if "No laptops match the criteria" in response:
            return (
                "No laptops match the exact criteria. "
                "Try refining your query or removing strict constraints (e.g., price)."
            )
        if isinstance(response, dict) and 'output' in response:
            return response['output'] 
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if user_query:
    st.markdown("##### Recommended Laptops")
    try:
        # Replace with your recommendation logic or LLM-based function
        recommendations = get_gemini_recommendations(user_query, data)  # Function definition in your LLM integration
        st.markdown(recommendations)
    except Exception as e:
        st.error("Error generating recommendations.")
        st.exception(e)
        
# Sidebar filters
st.sidebar.title("Filter Options")
selected_clusters = st.sidebar.multiselect("Select Laptop Functionality", options=data['cluster_label'].unique(), default=data['cluster_label'].unique())
selected_brands = st.sidebar.multiselect("Select Brand(s)", options=data['brand'].unique(), default=data['brand'].unique())
selected_processor = st.sidebar.multiselect("Select Processor(s)", options=data['processor_brand'].unique(), default=data['processor_brand'].unique())
selected_ram = st.sidebar.multiselect("Select RAM Size", options=data['ram_size'].unique(), default=data['ram_size'].unique())
price_range = st.sidebar.slider("Select Price Range (in Rp.)", int(data['price(in Rp.)'].min()), int(data['price(in Rp.)'].max()), (int(data['price(in Rp.)'].min()), int(data['price(in Rp.)'].max())))

# Filter data
filtered_data = data[
    (data['brand'].isin(selected_brands)) &
    (data['processor_brand'].isin(selected_processor)) &
    (data['ram_size'].isin(selected_ram)) &
    (data['price(in Rp.)'] >= price_range[0]) &
    (data['price(in Rp.)'] <= price_range[1]) &
    (data['cluster_label'].isin(selected_clusters))
]

# Dashboard title and quick summary
st.markdown("### Quick Summary")
brand_counts = filtered_data['brand'].value_counts()
processor_counts = filtered_data['processor_brand'].value_counts()
average_price = f"Rp {filtered_data['price(in Rp.)'].mean():,.0f}"
highest_average_rating_brand = filtered_data.groupby('brand')['rating'].mean().idxmax()
highest_average_rating = filtered_data.groupby('brand')['rating'].mean().max()

# Render Quick Summary
st.markdown(
    f"""
    <table style="width: 100%; font-size: 16px; border-collapse: collapse;">
        <tr><td><b>Most Popular Brand</b></td><td>{brand_counts.idxmax()}</td></tr>
        <tr><td><b>Average Price</b></td><td>{average_price}</td></tr>
        <tr><td><b>Most Popular Processor</b></td><td>{processor_counts.idxmax()}</td></tr>
        <tr><td><b>Highest Average Rating Brand</b></td><td>{highest_average_rating_brand} ({highest_average_rating:.2f})</td></tr>
    </table>
    """,
    unsafe_allow_html=True,
)

# Visualizations
st.markdown("### Visualizations")
# Brand Distribution
brand_distribution = filtered_data['brand'].value_counts().reset_index()
brand_distribution.columns = ['brand', 'count']
fig1 = px.pie(brand_distribution, values='count', names='brand', title="Brand Distribution")
st.plotly_chart(fig1)

# Average Price by Brand
avg_price = filtered_data.groupby('brand')['price(in Rp.)'].mean().reset_index()
fig2 = px.bar(avg_price, x='brand', y='price(in Rp.)', title="Average Price by Brand", text='price(in Rp.)', color='brand')
fig2.update_traces(texttemplate='Rp %{text:,.0f}', textposition='outside')
fig2.update_layout(xaxis_title="Brand", yaxis_title="Average Price (in Rp.)", showlegend=False)
st.plotly_chart(fig2)

# Average Ratings by Brand
avg_rating = filtered_data.groupby('brand')['rating'].mean().reset_index()
fig3 = px.bar(avg_rating, x='brand', y='rating', title="Average Ratings by Brand", text='rating', color='brand')
fig3.update_traces(texttemplate='%{text:.1f}', textposition='outside')
fig3.update_layout(xaxis_title="Brand", yaxis_title="Average Ratings", showlegend=False)
st.plotly_chart(fig3)

# Distribution of Processor Brands
processor_distribution = filtered_data['processor_brand'].value_counts().reset_index()
processor_distribution.columns = ['processor_brand', 'count']
fig4 = px.pie(processor_distribution, values='count', names='processor_brand', title="Processor Brand Distribution")
st.plotly_chart(fig4)

# Price vs Rating Scatter Plot
fig5 = px.scatter(
    filtered_data,
    x='price(in Rp.)',
    y='rating',
    color='brand',
    hover_data=['name', 'processor', 'ram', 'storage'],
    title="Price vs Rating"
)
fig5.update_traces(marker=dict(size=8))  # Uniform marker size
fig5.update_layout(xaxis_title="Price (in Rp.)", yaxis_title="Rating")
st.plotly_chart(fig5)

# Price vs Rating by Laptop Functionality
fig_functionality = px.scatter(
    filtered_data,
    x='price(in Rp.)',
    y='rating',
    color='cluster_label',
    hover_data=['name', 'processor', 'ram_size', 'storage_size', 'storage_type'],
    title="Price vs Rating by Laptop Functionality",
    labels={'cluster_label': 'Functionality'},
)
fig_functionality.update_traces(marker=dict(size=10, opacity=0.8))
fig_functionality.update_layout(
    xaxis_title="Price (in Rp.)",
    yaxis_title="Rating",
    legend_title="Laptop Functionality"
)
st.plotly_chart(fig_functionality)

# Footer
st.markdown("---")
st.markdown("**Created by Dwina Agustin Putri**")
