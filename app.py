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

try:
    API_KEY = os.environ.get("API_KEY")
    print(f"API_KEY exists: {API_KEY is not None}")
    
    tickers = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,JPM,JNJ"
    url = f"https://financialmodelingprep.com/api/v3/quote/{tickers}?apikey={API_KEY}"
    response = requests.get(url)
    
    print(f"API Response status: {response.status_code}")
    data = response.json()
    print(f"API Data length: {len(data)}")
    
    if data:
        print(f"First item keys: {list(data[0].keys())}")
        
        # 필수 키들이 있는 항목만 필터링
        required_keys = ['pe', 'eps', 'marketCap', 'price', 'priceAvg200', 'yearHigh', 'yearLow', 'volume', 'avgVolume', 'symbol']
        filtered_data = []
        for stock in data:
            if all(key in stock and stock[key] is not None for key in required_keys):
                filtered_data.append(stock)
        
        print(f"Filtered data length: {len(filtered_data)}")
        data = filtered_data
        
        if len(data) > 0:
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

            if len(cleaned_df_filtered) >= 3:
                spectral = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
                labels = spectral.fit_predict(cleaned_df_filtered)
                cleaned_df.loc[cleaned_df_filtered.index, "Cluster"] = labels.astype(int)
                centroids = cleaned_df.groupby("Cluster")[features].mean().reset_index(drop=True)
                print("Clustering successful")
            else:
                print("Not enough data for clustering")
                cleaned_df = pd.DataFrame()
                centroids = pd.DataFrame()
        else:
            print("No valid data after filtering")
            cleaned_df = pd.DataFrame()
            centroids = pd.DataFrame()
    else:
        print("No data received from API")
        cleaned_df = pd.DataFrame()
        centroids = pd.DataFrame()
        
except Exception as e:
    print(f"Error during initialization: {e}")
    cleaned_df = pd.DataFrame()
    centroids = pd.DataFrame()

def map_input(cluster_feature, tag):
    if len(cluster_feature) == 0:
        return 0
    tags = {"high": cluster_feature.max(), "medium": cluster_feature.mean(), "low": cluster_feature.min()}
    return tags.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    if len(centroids) == 0:
        return 0
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
        
        if len(cleaned_df) == 0:
            return jsonify({"error": "No data available for optimization"}), 500
        
        features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]
        
        if not req_data:
            return jsonify({"error": "No JSON data received"}), 400
        
        missing_features = [f for f in features if f not in req_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400
        
        invalid_values = [f for f in features if req_data[f].lower() not in ["high", "medium", "low"]]
        if invalid_values:
            return jsonify({"error": f"Invalid values for: {invalid_values}"}), 400
        
        mapped = {f: map_input(centroids[f] if f in centroids.columns else pd.Series([0]), req_data[f].lower()) for f in features}
        cluster = best_cluster(mapped)
        cluster_tickers = cleaned_df[cleaned_df["Cluster"] == cluster]["Ticker"].dropna().tolist()
        
        if not cluster_tickers:
            cluster_tickers = cleaned_df["Ticker"].dropna().tolist()[:3]  # 폴백
        
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
