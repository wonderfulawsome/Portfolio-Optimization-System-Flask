import os
from math import sqrt
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get('API_KEY')
TICKERS = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,JPM,JNJ"
features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]

def fetch_stock_data():
    url = f"https://financialmodelingprep.com/api/v3/quote/{TICKERS}?apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    price_diffs = [(stock['price'] - stock['priceAvg200']) for stock in data]
    price_ranges = [(stock['yearHigh'] - stock['yearLow']) for stock in data]
    volume_ratios = [(stock['volume'] / stock['avgVolume']) for stock in data]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(np.column_stack([price_diffs, price_ranges, volume_ratios]))
    
    return pd.DataFrame([{
        'symbol': stock['symbol'],
        'pe': stock['pe'],
        'eps': stock['eps'],
        'marketCap': stock['marketCap'],
        'norm_price_diffs': scaled_features[i, 0],
        'norm_price_ranges': scaled_features[i, 1],
        'norm_volume_ratios': scaled_features[i, 2]
    } for i, stock in enumerate(data)])

def fetch_historical_data():
    historical_data = {}
    for ticker in TICKERS.split(','):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=252&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if isinstance(data, list) and data:
            prices = [day['close'] for day in data]
            historical_data[ticker] = prices
    
    price_df = pd.DataFrame(historical_data)
    return price_df.pct_change().dropna()

stock_df = fetch_stock_data()
stock_filtered = stock_df[features].dropna()

spectral = SpectralClustering(n_clusters=3, affinity='rbf')
labels = spectral.fit_predict(stock_filtered)
stock_df.loc[stock_filtered.index, "Cluster"] = labels.astype(int)
centroids = stock_df.groupby("Cluster")[features].mean().reset_index(drop=True)

def map_input(cluster_feature, tag):
    tags = {"high": cluster_feature.max(), "medium": cluster_feature.mean(), "low": cluster_feature.min()}
    return tags.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    dist = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dist.idxmin())

def stats(weights, mean_ret, cov):
    ret = np.dot(weights, mean_ret)
    vol = sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sr = (ret - 0.01) / vol
    return ret, vol, sr

@app.route("/")
def index():
    return jsonify({"status": "running", "features": features})

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.json
    if any(f not in data for f in features):
        return jsonify({"error": "Missing features"}), 400
    if any(data[f].lower() not in ["high", "medium", "low"] for f in features):
        return jsonify({"error": "Invalid values"}), 400
    
    mapped = {f: map_input(centroids[f], data[f].lower()) for f in features}
    cluster = best_cluster(mapped)
    tickers = stock_df[stock_df["Cluster"] == cluster]["symbol"].dropna().tolist()
    
    if not tickers:
        return jsonify({"error": "No tickers found"}), 400
    
    returns = fetch_historical_data()
    cluster_returns = returns[[col for col in returns.columns if col in tickers]]
    
    if cluster_returns.empty:
        return jsonify({"error": "No historical data"}), 400
    
    mean_ret = cluster_returns.mean() * 252
    cov = cluster_returns.cov() * 252
    
    weights = np.random.dirichlet(np.ones(len(mean_ret)), 10000)
    res = np.array([stats(w, mean_ret, cov) for w in weights])
    best = res[:, 2].argmax()
    
    return jsonify({
        "closest_cluster": cluster,
        "optimized_companies": tickers,
        "optimal_portfolio": {t: round(w * 100, 2) for t, w in zip(mean_ret.index, weights[best])},
        "expected_return": float(res[best, 0]),
        "expected_volatility": float(res[best, 1]),
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
