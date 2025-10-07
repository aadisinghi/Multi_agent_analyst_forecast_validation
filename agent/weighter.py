import json
import yfinance as yf
import pandas as pd

with open("ds_outputs/stock_list.json", "r", encoding="utf-8") as f:
    stock_list = json.load(f)

results = []
brokerage_scores = {}

for entry in stock_list:
    ticker = entry.get("ticker")
    brokerage = entry.get("brokerage")
    issue_date_str = entry.get("date")
    term = entry.get("term")

    if not ticker or not issue_date_str:
        continue

    try:
        issue_date = pd.to_datetime(issue_date_str)
    except Exception:
        continue

    df = yf.download(ticker, start=issue_date.strftime("%Y-%m-%d"))
    if df.empty:
        continue

    price_on_issue = df.iloc[0]["Close"]
    if isinstance(price_on_issue, pd.Series):
        price_on_issue = price_on_issue.iloc[0]
    price_on_issue = float(price_on_issue)

    price_now = df.iloc[-1]["Close"]
    if isinstance(price_now, pd.Series):
        price_now = price_now.iloc[0]
    price_now = float(price_now)
    
    pct_change = ((price_now - price_on_issue) / price_on_issue) * 100

    results.append({
        "name": entry.get("name"),
        "ticker": ticker,
        "brokerage": brokerage,
        "issue_date": issue_date_str,
        "term": term,
        "price_on_issue": price_on_issue,
        "price_now": price_now,
        "pct_change": pct_change
    })

    if brokerage:
        if brokerage not in brokerage_scores:
            brokerage_scores[brokerage] = {"total": 0, "sum_pct": 0}
        brokerage_scores[brokerage]["total"] += 1
        brokerage_scores[brokerage]["sum_pct"] += pct_change

pd.DataFrame(results).to_csv("ds_outputs/brokerage_performance_results.csv", index=False)

print("Brokerage average returns:")