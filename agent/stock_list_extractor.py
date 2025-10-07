import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import time 

"""
This script uses DeepSeek's API to extract stock names , tickers and fund manager from trading headlines.
"""

load_dotenv("keys.env")
api_key_deepseek = os.getenv("DEEPSEEK")
client = OpenAI(api_key=api_key_deepseek, base_url="https://api.deepseek.com")

# Load headlines/signals
with open("scraper_outputs/testing_stock_recos.json", "r", encoding="utf-8") as f:
    signals = json.load(f)

# BATCH_SIZE = 50 
# def chunk_list(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# Prompt DeepSeek to extract stock names and tickers from headlines
system_prompt = """
    You are a financial analyst specializing in Indian equities.
    Analyze the following trading headlines and extract a list of all relevant stock names , their official NSE tickers , which fund manager is recommending the stock, what the action is: buy/sell, the time horizon: long term/short term/ medium/ any specific days or months.
    If a headline mentions multiple stocks, include all.
    You are also supposed to scan the URL provided and analyze whats written in the article to extract the different stock names and brokerage firms and most importantly the time horizon.
    Return your output as a JSON array, where each element contains 'name', 'ticker', 'brokerage', 'date', 'term'.
    For example, a headline exists as follows:{
    title: Stocks to buy in 2025 for long term: Suzlon, Inox Wind among 5 stocks that could give 12-36% return,
    date: Sep 3, 2025, 04:41 PM IST,
    summary: Brokerages remain upbeat on select stocks across renewables, infrastructure, and real estate, projecting healthy upsides in the near to medium term.We have collated a list of recommendations from top brokerage firms from ETNow and other sources:,
    url: https://economictimes.indiatimes.com/markets/stocks/recos/stocks-to-buy-in-2025-for-long-term-suzlon-inox-wind-among-5-stocks-that-could-give-12-36-return/slideshow/123667876.cms,
    your task is to give me [name: Suzlon Energy, ticker: SUZLON.NS, brokerage: XYZ Mutual Fund, date: Sep 3, 2025, term: Medium], [name: Inox Wind, ticker: INOXWIND.NS, brokerage: ABC Capital, date: Sep 3, 2025, term: Medium].
    It is crucial that you scan the URL provided and analyze whats written in the article to extract the different stock names , brokerage firms and make sure the stock ticker is correct. 
    Also, ensure that the rationale is captured properly. It might not be mentioned in the url, so just search the brokerage name and the stock name. 
    if there are multiple brokerages recommending the same stock, include all of them as separate entries. 
    if the name of all the multiple brokerages are not provided , just put "Multiple brokerages".
    if the name of the brokerage is not provided, just put "Unknown".

    return a json file as follows:
    [
        {
            "name": "Suzlon Energy",
            "ticker": "SUZLON.NS",
            "brokerage": "XYZ Mutual Fund",
            "date": "Sep 3, 2025",
            "term": "Medium",
        },
        {
            "name": "Inox Wind",
            "ticker": "INOXWIND.NS",
            "brokerage": "ABC Capital",
            "date": "Sep 3, 2025",
            "term": "Medium",
        }
    ] 

    Make sure you analyze the complete file provided and return all the extractions in the format provided,especially the time horizon after analyzing the URLs.
    """

# all_results = [] 

# for i, chunk in enumerate(chunk_list(signals, BATCH_SIZE)):
#     print(f"Processing chunk {i+1}/{(len(signals) + BATCH_SIZE - 1) // BATCH_SIZE}...")
# prompt = (
#     "Here are the trading headlines:\n\n"
#     f"{json.dumps(chunk, indent=2)}"
# )

prompt = (
    "Here are the trading headlines:\n\n"
    f"{json.dumps(signals, indent=2)}"
)

print("Prompt prepared, sending to DeepSeek...")
start_time  = time.time()
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ],
    stream=False
) 

end_time = time.time() 
print(f"DeepSeek API call completed in {end_time - start_time:.2f} seconds.")

    # try:
    #     chunk_result = json.loads(response.choices[0].message.content)
    #     all_results.extend(chunk_result)
    # except Exception as e:
    #     print(f"Error parsing chunk {i+1}: {e}")

print("Response received from DeepSeek.")

with open("ds_outputs/stock_list.json", "w", encoding="utf-8") as f:
    # json.dump(all_results, f, indent=2, ensure_ascii=False)
    json.dump(response.choices[0].message.content, f, indent=2)

print("Extracted stock list saved to ds_outputs/stock_list.json") 