import json
from datetime import datetime
from openai import OpenAI 
from dotenv import load_dotenv
import os 

"""
This script uses DeepSeek's API to analyze trading signals and provide buy/sell recommendations.
"""

load_dotenv()
api_key_deepseek = os.getenv("DEEPSEEK")
client = OpenAI(api_key=api_key_deepseek, base_url="https://api.deepseek.com")

# Load signals
with open("data_json\stock_recos_demo_1.json", "r", encoding="utf-8") as f:
    signals = json.load(f)

system_prompt = "You are a professional financial analyst specializing in Indian equities and swing trading. " \
"For each trading signal provided, visit the associated URL and conduct a thorough analysis. " \
"Your output should include: " \
"(1) clear buy/sell recommendations, " \
"(2) suggested stop loss and target price, " \
"(3) estimated risk and reward, " \
"(4) recommended time horizon, and " \
"(5) a concise rationale for your recommendation based on both the signal and your own assessment of the stock's fundamentals and technicals. " \
"Ensure your analysis is detailed, actionable, and tailored for swing traders."


prompt = ("Manually analyze the following trading signals scraped from ettimes.com. "
    "Go through each signal and provide your recommendations and reasoning. "
    "Here is the data:\n\n"
    f"{json.dumps(signals, indent=2)}")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a professional financial analyst specializing in Indian equities and swing trading. For each trading signal provided, visit the associated URL and conduct a thorough analysis. Your output should include: (1) clear buy/sell recommendations, (2) suggested stop loss and target price, (3) estimated risk and reward, (4) recommended time horizon, and (5) a concise rationale for your recommendation based on both the signal and your own assessment of the stock's fundamentals and technicals. Ensure your analysis is detailed, actionable, and tailored for swing traders."},
        {"role": "user", "content": prompt},
    ],
    stream=False
)

output = response.choices[0].message.content

with open("deepseek_response.json", "w", encoding="utf-8") as f:
    f.write(output)

print("Response saved to deepseek_response.json")