# Stock Analyzer App (Streamlit Version)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import datetime
import numpy as np

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

sector_averages = {
    "Technology": {"P/E Ratio": 25, "P/S Ratio": 5, "P/B Ratio": 6},
    "Healthcare": {"P/E Ratio": 20, "P/S Ratio": 4, "P/B Ratio": 3},
    "Financials": {"P/E Ratio": 15, "P/S Ratio": 2, "P/B Ratio": 1.5},
    "Energy": {"P/E Ratio": 12, "P/S Ratio": 1.2, "P/B Ratio": 1.3},
}

def query_mistral(question):
    payload = {"inputs": question, "parameters": {"max_length": 256}}
    response = requests.post(API_URL, headers=headers, json=payload)
    output = response.json()
    return output[0].get("generated_text", "No response from model.")

def calculate_ratios(market_cap, total_revenue, price, dividend_amount, eps, growth, book_value):
    pe = price / eps if eps else 0
    ps = market_cap / total_revenue if total_revenue else 0
    pb = market_cap / book_value if book_value else 0
    peg = pe / (growth * 100) if growth else 0
    div_yield = (dividend_amount / price) * 100 if price else 0
    return {
        'P/E Ratio': pe,
        'P/S Ratio': ps,
        'P/B Ratio': pb,
        'PEG Ratio': peg,
        'Dividend Yield': div_yield
    }

def stock_research(symbol, eps, growth, book_value):
    info = {"Name": symbol, "Industry": "Tech", "Sector": "Technology", "Market Cap": np.random.randint(1e9, 3e9)}
    price = np.random.uniform(100, 300)
    dividend = np.random.uniform(0, 5)
    dates = pd.date_range(datetime.date.today() - datetime.timedelta(days=365), periods=365)
    prices = np.random.uniform(100, 300, size=365)
    smooth_prices = np.convolve(prices, np.ones(5)/5, mode='valid')

    ratios = calculate_ratios(info['Market Cap'], info['Market Cap']/5, price, dividend, eps, growth, book_value)
    summary_prompt = f"Summarize this report:\nName: {symbol}\nSector: {info['Sector']}\n" + \
                     '\n'.join([f"{k}: {round(v, 2)}" for k, v in ratios.items()])
    summary = query_mistral(summary_prompt)

    return info, ratios, dates[:len(smooth_prices)], smooth_prices, summary

# Streamlit UI
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("📈 Stock Analyzer")

symbol = st.text_input("Stock Symbol", value="AAPL")
eps = st.number_input("Earnings Per Share (EPS)", value=5.0)
growth = st.number_input("Growth Rate", value=0.1)
book_value = st.number_input("Book Value", value=500000000)

if st.button("Run Analysis"):
    info, ratios, dates, prices, summary = stock_research(symbol, eps, growth, book_value)

    st.subheader("📋 Company Info")
    st.dataframe(pd.DataFrame(info.items(), columns=["Metric", "Value"]))

    st.subheader("📊 Valuation Ratios")
    st.dataframe(pd.DataFrame(ratios.items(), columns=["Ratio", "Value"]))

    st.subheader("🧠 AI Summary")
    st.write(summary)

    st.subheader("📉 Historical Price Chart")
    fig, ax = plt.subplots()
    ax.plot(dates, prices)
    ax.set_title(f"{symbol} Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)
