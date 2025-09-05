import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# 로그 출력 강제
import sys
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print("=== APPLICATION STARTING ===")
print(f"API_KEY exists: {'API_KEY' in os.environ}")
print(f"Environment variables: {list(os.environ.keys())}")

# 더미 데이터로 시작
features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]

# 간단한 더미 데이터
stock_data = pd.DataFrame({
    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'pe': [25.0, 30.0, 28.0],
    'eps': [6.0, 8.0, 5.0], 
    'marketCap': [3000000000000, 2500000000000, 1800000000000],
    'norm_price_diffs': [0.5, -0.2, 0.1],
    'norm_price_ranges': [1.2, 0.8, 1.0], 
    'norm_volume_ratios': [-0.3, 0.4, 0.0],
    'Cluster': [0, 1, 2]
})

centroids = stock_data.groupby("Cluster")[features].mean().reset_index(drop=True)

print("=== DATA INITIALIZED ===")
print(f"Stock data shape: {stock_data.shape}")
print(f"Centroids shape: {centroids.shape}")

def map_input(cluster_feature, tag):
    tags = {"high": cluster_feature.max(), "medium": cluster_feature.mean(), "low": cluster_feature.min()}
    return tags.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    dist = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dist.idxmin())

@app.route("/")
def index():
    return jsonify({"status": "Portfolio Optimization API is running!", "features": features})

@app.route("/test")
def test():
    return jsonify({
        "message": "Test endpoint working",
        "api_key_exists": bool(os.environ.get('API_KEY')),
        "features": features,
        "stock_count": len(stock_data)
    })

@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        print("=== OPTIMIZE REQUEST RECEIVED ===")
        
        data = request.json
        print(f"Request data: {data}")
        
        if not data:
            print("ERROR: No JSON data")
            return jsonify({"error": "No JSON data provided"}), 400
            
        # 요청 데이터 검증
        missing = [f for f in features if f not in data]
        if missing:
            print(f"ERROR: Missing features: {missing}")
            return jsonify({"error": f"Missing features: {missing}"}), 400
            
        invalid = [f for f in features if data[f].lower() not in ["high", "medium", "low"]]
        if invalid:
            print(f"ERROR: Invalid values: {invalid}")
            return jsonify({"error": f"Invalid values for: {invalid}"}), 400

        # 클러스터 선택
        mapped = {f: map_input(centroids[f], data[f].lower()) for f in features}
        cluster = best_cluster(mapped)
        cluster_symbols = stock_data[stock_data["Cluster"] == cluster]["symbol"].tolist()
        
        print(f"Selected cluster: {cluster}")
        print(f"Cluster symbols: {cluster_symbols}")

        # 더미 포트폴리오 생성
        portfolio = {symbol: round(100/len(cluster_symbols), 2) for symbol in cluster_symbols}
        
        result = {
            "closest_cluster": cluster,
            "optimized_companies": cluster_symbols,
            "optimal_portfolio": portfolio,
            "expected_return": 0.12,
            "expected_volatility": 0.18,
            "message": "Using dummy data for testing"
        }
        
        print(f"SUCCESS: {result}")
        return jsonify(result), 200
        
    except Exception as e:
        print(f"ERROR in optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    print("=== STARTING FLASK SERVER ===")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
