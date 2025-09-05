import os
from math import sqrt
import numpy as np
import pandas as pd
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

# 하드코딩된 테스트 데이터
test_data = [
    {'symbol': 'AAPL', 'pe': 25.5, 'eps': 6.05, 'marketCap': 3000000000000, 'price': 150, 'priceAvg200': 145, 'yearHigh': 180, 'yearLow': 120, 'volume': 50000000, 'avgVolume': 45000000},
    {'symbol': 'MSFT', 'pe': 30.2, 'eps': 8.12, 'marketCap': 2800000000000, 'price': 300, 'priceAvg200': 295, 'yearHigh': 350, 'yearLow': 250, 'volume': 30000000, 'avgVolume': 28000000},
    {'symbol': 'GOOGL', 'pe': 22.8, 'eps': 4.56, 'marketCap': 1700000000000, 'price': 120, 'priceAvg200': 115, 'yearHigh': 140, 'yearLow': 90, 'volume': 25000000, 'avgVolume': 23000000},
    {'symbol': 'AMZN', 'pe': 45.1, 'eps': 2.31, 'marketCap': 1500000000000, 'price': 140, 'priceAvg200': 135, 'yearHigh': 170, 'yearLow': 110, 'volume': 35000000, 'avgVolume': 32000000},
    {'symbol': 'TSLA', 'pe': 65.3, 'eps': 3.22, 'marketCap': 800000000000, 'price': 250, 'priceAvg200': 240, 'yearHigh': 300, 'yearLow': 180, 'volume': 40000000, 'avgVolume': 38000000}
]

try:
    data = test_data
    print(f"Using test data, length: {len(data)}")
    
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

    spectral = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
    labels = spectral.fit_predict(cleaned_df_filtered)
    cleaned_df.loc[cleaned_df_filtered.index, "Cluster"] = labels.astype(int)
    centroids = cleaned_df.groupby("Cluster")[features].mean().reset_index(drop=True)
    print("Initialization successful")

except Exception as e:
    print(f"Error during initialization: {e}")
    cleaned_df = pd.DataFrame()
    centroids = pd.DataFrame()

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
    try:
        req_data = request.json
        print("Received data:", req_data)
        
        features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]
        
        if not req_data:
            return jsonify({"error": "No JSON data received"}), 400
        
        missing_features = [f for f in features if f not in req_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        mapped = {f: map_input(centroids[f], req_data[f].lower()) for f in features}
        cluster = best_cluster(mapped)
        cluster_tickers = cleaned_df[cleaned_df["Cluster"] == cluster]["Ticker"].dropna().tolist()
        
        if not cluster_tickers:
            cluster_tickers = ["AAPL", "MSFT", "GOOGL"]
        
        portfolio_weights = np.random.dirichlet(np.ones(len(cluster_tickers)))
        portfolio = {ticker: round(weight * 100, 2) for ticker, weight in zip(cluster_tickers, portfolio_weights)}
        
        return jsonify({
            "closest_cluster": cluster,
            "optimized_companies": cluster_tickers,
            "optimal_portfolio": portfolio,
            "expected_return": 0.08,
            "expected_volatility": 0.15,
        })
    
    except Exception as e:
        print("Error in optimize:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
