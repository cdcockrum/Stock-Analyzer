import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import requests
import os
import datetime
import tempfile
import numpy as np

# Your Hugging Face API Token
HF_Token = os.getenv("HF_Token")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

headers = {
    "Authorization": f"Bearer {HF_Token}"
}

def query_mistral(question):
    payload = {"inputs": question, "parameters": {"max_length": 256}}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    try:
        output = response.json()
        # Check for standard output format
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        else:
            # Return error message or full object for debugging
            return f"[Error from Mistral API]: {output}"
    except Exception as e:
        return f"[Exception in query_mistral]: {str(e)}"


POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

sector_averages = {
    "Technology": {"P/E Ratio": 25, "P/S Ratio": 5, "P/B Ratio": 6},
    "Healthcare": {"P/E Ratio": 20, "P/S Ratio": 4, "P/B Ratio": 3},
    "Financials": {"P/E Ratio": 15, "P/S Ratio": 2, "P/B Ratio": 1.5},
    "Energy": {"P/E Ratio": 12, "P/S Ratio": 1.2, "P/B Ratio": 1.3},
}

def safe_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response
    except:
        return None

def calculate_ratios(market_cap, total_revenue, price, dividend_amount, eps=5.0, growth=0.1, book_value=500000000):
    pe = price / eps if eps else 0
    ps = market_cap / total_revenue if total_revenue else 0
    pb = market_cap / book_value if book_value else 0
    peg = pe / (growth * 100) if growth else 0
    div_yield = (dividend_amount / price) * 100 if price else 0
    debt_equity = np.random.uniform(0.2, 2.0)
    roe = np.random.uniform(5, 25)
    free_cash_flow = np.random.uniform(50000000, 500000000)
    beta = np.random.uniform(0.8, 1.5)
    ev_ebitda = np.random.uniform(8, 20)  # Placeholder random value
    price_cash_flow = np.random.uniform(10, 25)
    operating_margin = np.random.uniform(10, 30)
    revenue_growth = np.random.uniform(5, 20)
    return {
        'P/E Ratio': pe,
        'P/S Ratio': ps,
        'P/B Ratio': pb,
        'PEG Ratio': peg,
        'Dividend Yield': div_yield,
        'Debt/Equity Ratio': debt_equity,
        'Return on Equity (%)': roe,
        'Free Cash Flow ($)': free_cash_flow,
        'Beta (Volatility)': beta,
        'EV/EBITDA': ev_ebitda,
        'Price/Cash Flow': price_cash_flow,
        'Operating Margin (%)': operating_margin,
        'Revenue Growth (%)': revenue_growth
    }

def stock_research(symbol, eps=5.0, growth=0.1, book=500000000):
    info = {"Name": symbol, "Industry": "Tech", "Sector": "Technology", "Market Cap": np.random.randint(1000000000, 3000000000)}
    price = np.random.uniform(100, 300)
    dividends = np.random.uniform(0, 5)
    dates = pd.date_range(datetime.date.today() - datetime.timedelta(days=365), periods=365)
    prices = np.random.uniform(100, 300, size=365)

    ratios = calculate_ratios(info['Market Cap'], info['Market Cap']/5, price, dividends, eps, growth, book)
    ratios = {k: round(v, 2) for k, v in ratios.items()}

    sector_comp = pd.DataFrame({"Metric": ["Example"], "Value": [0]})

    smooth_prices = np.convolve(prices, np.ones(5)/5, mode='valid')
    fig, ax = plt.subplots()
    ax.plot(dates[:len(smooth_prices)], smooth_prices)
    ax.set_title(f"{symbol} Historical Price (Smoothed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.grid(True)

    info_table = pd.DataFrame(info.items(), columns=["Metric", "Value"])
    ratios_table = pd.DataFrame(ratios.items(), columns=["Metric", "Value"])

    financial_health_metrics = [
        "Debt/Equity Ratio", "Return on Equity (%)", "Free Cash Flow ($)", "Beta (Volatility)"
    ]
    financial_health = ratios_table[ratios_table["Metric"].isin(financial_health_metrics)]

    recommendation = "Hold"
    if ratios['P/E Ratio'] < 15 and ratios['Debt/Equity Ratio'] < 1.0 and ratios['Return on Equity (%)'] > 10 and ratios['Beta (Volatility)'] < 1.2:
        recommendation = "Buy"
    elif ratios['P/E Ratio'] > 30 or ratios['Debt/Equity Ratio'] > 2.0 or ratios['Return on Equity (%)'] < 5:
        recommendation = "Sell"

    report = (
        f"Company Overview:\n"
        f"Name: {info.get('Name', 'N/A')}\n"
        f"Industry: {info.get('Industry', 'N/A')}\n"
        f"Sector: {info.get('Sector', 'N/A')}\n"
        f"Market Cap: ${info.get('Market Cap', 0):,.2f}\n\n"
        f"Financial Metrics:\n"
        f"P/E Ratio: {ratios.get('P/E Ratio', 'N/A')}\n"
        f"P/S Ratio: {ratios.get('P/S Ratio', 'N/A')}\n"
        f"P/B Ratio: {ratios.get('P/B Ratio', 'N/A')}\n"
        f"PEG Ratio: {ratios.get('PEG Ratio', 'N/A')}\n"
        f"Dividend Yield: {ratios.get('Dividend Yield', 'N/A')}%\n"
        f"Debt/Equity Ratio: {ratios.get('Debt/Equity Ratio', 'N/A')}\n"
        f"Return on Equity: {ratios.get('Return on Equity (%)', 'N/A')}%\n"
        f"Free Cash Flow: ${ratios.get('Free Cash Flow ($)', 0):,.2f}\n"
        f"Beta (Volatility): {ratios.get('Beta (Volatility)', 'N/A')}\n"
        f"EV/EBITDA: {ratios.get('EV/EBITDA', 'N/A')}\n"
        f"Price/Cash Flow: {ratios.get('Price/Cash Flow', 'N/A')}\n"
        f"Operating Margin: {ratios.get('Operating Margin (%)', 'N/A')}%\n"
        f"Revenue Growth: {ratios.get('Revenue Growth (%)', 'N/A')}%\n"
    )

    summary_prompt = f"Summarize this financial report clearly and briefly:\n\n{report}"
    ai_summary = query_mistral(summary_prompt)

    financial_health = pd.concat([
        financial_health,
        pd.DataFrame([{"Metric": "Recommendation", "Value": recommendation}])
    ], ignore_index=True)

    return ai_summary, info_table, ratios_table, financial_health, sector_comp, fig
# Theme Selection
selected_theme = os.getenv("APP_THEME", "light")
if selected_theme == "dark":
    theme = gr.themes.Base()
else:
    theme = gr.themes.Soft(primary_hue="blue")


# Fetch Functions
def get_company_info(symbol):
    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}?apiKey={POLYGON_API_KEY}"
    response = safe_request(url)
    if response:
        data = response.json().get('results', {})
        sector = data.get('market', 'Technology')
        if sector.lower() == 'stocks':
            sector = 'Technology'
        return {
            'Name': data.get('name', 'N/A'),
            'Industry': data.get('sic_description', 'N/A'),
            'Sector': sector,
            'Market Cap': data.get('market_cap', 0),
            'Total Revenue': data.get('total_employees', 0) * 100000
        }
    return None

def get_current_price(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
    response = safe_request(url)
    if response:
        return response.json()['results'][0]['c']
    return None

def get_dividends(symbol):
    url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol}&apiKey={POLYGON_API_KEY}"
    response = safe_request(url)
    if response:
        return response.json()['results'][0].get('cash_amount', 0)
    return 0

def get_historical_prices(symbol):
    end = datetime.date.today()
    start = end - datetime.timedelta(days=365)
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    response = safe_request(url)
    if response:
        results = response.json()['results']
        dates = [datetime.datetime.fromtimestamp(r['t']/1000) for r in results]
        prices = [r['c'] for r in results]
        return dates, prices
    return [], []

# Financial Calculations
def calculate_ratios(market_cap, total_revenue, price, dividend_amount, eps=5.0, growth=0.1, book_value=500000000):
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

def compare_to_sector(sector, ratios):
    if sector.lower() == 'stocks':
        sector = 'Technology'
    averages = sector_averages.get(sector, {})
    if not averages:
        return pd.DataFrame({"Metric": ["Sector data not available"], "Value": ["N/A"]})

    data = {
        "Ratio": [],
        "Stock Value": [],
        "Sector Average": [],
        "Difference": []
    }
    for key in averages:
        stock_value = ratios.get(key, 0)
        sector_value = averages.get(key, 0)
        diff = stock_value - sector_value

        # Add emoji based on difference
        if diff < 0:
            diff_display = f"{diff:.2f} 🟢"
        elif diff > 0:
            diff_display = f"{diff:.2f} 🔴"
        else:
            diff_display = f"{diff:.2f} ⚪"

        data["Ratio"].append(key)
        data["Stock Value"].append(round(stock_value, 2))
        data["Sector Average"].append(round(sector_value, 2))
        data["Difference"].append(diff_display)

    return pd.DataFrame(data)

def generate_summary(info, ratios):
    recommendation = "Hold"
    if ratios['P/E Ratio'] < 15 and ratios['P/B Ratio'] < 2 and ratios['PEG Ratio'] < 1.0 and ratios['Dividend Yield'] > 2:
        recommendation = "Buy"
    elif ratios['P/E Ratio'] > 30 and ratios['P/B Ratio'] > 5 and ratios['PEG Ratio'] > 2.0:
        recommendation = "Sell"
    
    report = (
        f"Company Overview:\n"
        f"Name: {info['Name']}\n"
        f"Industry: {info['Industry']}\n"
        f"Sector: {info['Sector']}\n"
        f"Market Cap: ${info['Market Cap']:,.2f}\n\n"
        f"Financial Metrics:\n"
        f"P/E Ratio: {ratios['P/E Ratio']:.2f}\n"
        f"P/S Ratio: {ratios['P/S Ratio']:.2f}\n"
        f"P/B Ratio: {ratios['P/B Ratio']:.2f}\n"
        f"PEG Ratio: {ratios['PEG Ratio']:.2f}\n"
        f"Dividend Yield: {ratios['Dividend Yield']:.2f}%\n\n"
        f"Recommended Investment Action: {recommendation}.\n"
    )

    # Use Mistral to generate the summary
    summary_prompt = f"Summarize the following financial report clearly and briefly:\n\n{report}"
    return query_mistral(summary_prompt)



# Gradio UI
with gr.Blocks(theme=theme) as iface:
    with gr.Row():
        symbol = gr.Textbox(label="Stock Symbol (e.g., AAPL)")
        eps = gr.Number(label="Assumed EPS", value=5.0)
        growth = gr.Number(label="Assumed Growth Rate", value=0.1)
        book = gr.Number(label="Assumed Book Value", value=500000000)

    with gr.Tabs() as tabs:
        with gr.Tab("AI Research Summary"):
            output_summary = gr.Textbox()
        with gr.Tab("Company Snapshot"):
            output_info = gr.Dataframe()
        with gr.Tab("Valuation Ratios"):
            output_ratios = gr.Dataframe(label="Valuation Ratios")
        with gr.Tab("Financial Health"):
            output_health = gr.Dataframe()
        with gr.Tab("Sector Comparison"):
            output_sector = gr.Dataframe()
        with gr.Tab("Historical Price Chart"):
            output_chart = gr.Plot()
        with gr.Tab("Ask About Investing"):
            user_question = gr.Textbox(label="Ask about investing...")
            answer_box = gr.Textbox(label="Answer")
            ask_button = gr.Button("Get Answer")
            with gr.Row():
                ask_button.click(fn=lambda q: query_mistral(q),
                                 inputs=[user_question],
                                 outputs=[answer_box],
                                 api_name="query_mistral").then(
                    lambda: "",
                    inputs=[],
                    outputs=[user_question]
                )

    with gr.Row():
        submit_btn = gr.Button("Run Analysis")
        reset_btn = gr.Button("Reset All Fields")
        download_btn = gr.Button("Download Report")
        file_output = gr.File()

    submit_btn.click(fn=stock_research, inputs=[symbol, eps, growth, book],
                     outputs=[output_summary, output_info, output_ratios, output_health, output_sector, output_chart])


    def reset_fields():
        return "", 5.0, 0.1, 500000000, "", "", "", "", None
    
    reset_btn.click(
        fn=reset_fields,
        inputs=[],
        outputs=[
            symbol, eps, growth, book,
            output_summary, output_info,
            output_ratios, output_sector, output_chart
        ]
    )



    def reset_fields():
        return "", 5.0, 0.1, 500000000, "", "", "", "", None

    reset_btn.click(fn=reset_fields, inputs=[], outputs=[symbol, eps, growth, book, output_summary, output_info, output_ratios, output_sector, output_chart])



if __name__ == "__main__":
    iface.launch()
