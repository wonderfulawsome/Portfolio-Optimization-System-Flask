import os
from math import sqrt
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app, origins="*")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

API_KEY = os.environ.get("API_KEY")
tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,JPM,JNJ"
response = requests.get(f"https://financialmodelingprep.com/api/v3/quote/{tickers}?apikey={API_KEY}")
data = response.json()

data = [stock for stock in data if all(stock.get(key) is not None for key in ['pe', 'eps', 'marketCap', 'price', 'priceAvg200', 'yearHigh', 'yearLow', 'volume', 'avgVolume'])]

price_diffs = [(stock['price'] - stock['priceAvg200']) for stock in data]
price_ranges = [(stock['yearHigh'] - stock['yearLow']) for stock in data]
volume_ratios = [(stock['volume'] / stock['avgVolume']) for stock in data]

scaler = StandardScaler()
features_to_scale = np.column_stack([price_diffs, price_ranges, volume_ratios])
scaled_features = scaler.fit_transform(features_to_scale)

cleaned_df = pd.DataFrame({
    'Ticker': [stock['symbol'] for stock in data],
    'pe': [stock['pe'] for stock in data],
    'eps': [stock['eps'] for stock in data],
    'marketCap': [stock['marketCap'] for stock in data],
    'norm_price_diffs': scaled_features[:, 0],
    'norm_price_ranges': scaled_features[:, 1],
    'norm_volume_ratios': scaled_features[:, 2]
})

features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]
cleaned_df_filtered = cleaned_df[features].dropna()

spectral = SpectralClustering(n_clusters=3, affinity='rbf')
labels = spectral.fit_predict(cleaned_df_filtered)
cleaned_df.loc[cleaned_df_filtered.index, "Cluster"] = labels.astype(int)
centroids = cleaned_df.groupby("Cluster")[features].mean().reset_index(drop=True)

def map_input(cluster_feature, tag):
    tags = {"high": cluster_feature.max(), "medium": cluster_feature.mean(), "low": cluster_feature.min()}
    return tags.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    dist = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dist.idxmin())

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.json
    if any(f not in data for f in features):
        return jsonify({"error": "Missing features"}), 400
    if any(data[f].lower() not in ["high", "medium", "low"] for f in features):
        return jsonify({"error": "Values must be high / medium / low"}), 400
    
    mapped = {f: map_input(centroids[f], data[f].lower()) for f in features}
    cluster = best_cluster(mapped)
    cluster_tickers = cleaned_df[cleaned_df["Cluster"] == cluster]["Ticker"].dropna().tolist()
    
    portfolio_weights = np.random.dirichlet(np.ones(len(cluster_tickers)))
    portfolio = {ticker: round(weight * 100, 2) for ticker, weight in zip(cluster_tickers, portfolio_weights)}
    
    return jsonify({
        "closest_cluster": cluster,
        "optimized_companies": cluster_tickers,
        "optimal_portfolio": portfolio,
        "expected_return": 0.08,
        "expected_volatility": 0.15,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
