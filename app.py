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
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# API 설정
API_KEY = os.environ.get('API_KEY')
TICKERS = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,JPM,JNJ"

# 새로운 특성 리스트
features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]

def fetch_stock_data():
    """실시간 주식 데이터를 API에서 가져오기"""
    url = f"https://financialmodelingprep.com/api/v3/quote/{TICKERS}?apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    # 정규화를 위한 값들 계산
    price_diffs = [(stock['price'] - stock['priceAvg200']) for stock in data]
    price_ranges = [(stock['yearHigh'] - stock['yearLow']) for stock in data]
    volume_ratios = [(stock['volume'] / stock['avgVolume']) for stock in data]
    
    # 정규화
    scaler = StandardScaler()
    features_to_scale = np.column_stack([price_diffs, price_ranges, volume_ratios])
    scaled_features = scaler.fit_transform(features_to_scale)
    
    # DataFrame 생성
    stock_df = pd.DataFrame([{
        'symbol': stock['symbol'],
        'pe': stock['pe'],
        'eps': stock['eps'],
        'marketCap': stock['marketCap'],
        'norm_price_diffs': scaled_features[i, 0],
        'norm_price_ranges': scaled_features[i, 1],
        'norm_volume_ratios': scaled_features[i, 2],
        'price': stock['price']
    } for i, stock in enumerate(data)])
    
    return stock_df

def fetch_historical_data():
    """히스토리컬 데이터를 가져와서 수익률 계산"""
    historical_data = {}
    for ticker in TICKERS.split(','):
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=252&apikey={API_KEY}"
        response = requests.get(url)
        data = response.json()
        
        if 'historical' in data:
            prices = [day['close'] for day in data['historical'][::-1]]  # 역순으로 정렬
            historical_data[ticker] = prices
    
    # DataFrame으로 변환하고 수익률 계산
    price_df = pd.DataFrame(historical_data)
    returns = price_df.pct_change().dropna()
    
    return returns

# 앱 시작 시 데이터 로드 및 클러스터링
stock_data = fetch_stock_data()
stock_data_filtered = stock_data[features].dropna()

# Spectral Clustering
spectral = SpectralClustering(n_clusters=3, affinity='rbf')
labels = spectral.fit_predict(stock_data_filtered)
stock_data.loc[stock_data_filtered.index, "Cluster"] = labels.astype(int)
centroids = stock_data.groupby("Cluster")[features].mean().reset_index(drop=True)

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
    cluster_symbols = stock_data[stock_data["Cluster"] == cluster]["symbol"].dropna().tolist()

    if not cluster_symbols:
        return jsonify({"error": "No matching symbols in cluster"}), 400

    # 히스토리컬 데이터로 수익률 계산
    returns = fetch_historical_data()
    
    # 클러스터에 속한 종목들만 필터링
    cluster_returns = returns[[col for col in returns.columns if col in cluster_symbols]]
    
    if cluster_returns.empty:
        return jsonify({"error": "No historical data for cluster symbols"}), 400
    
    mean_ret = cluster_returns.mean() * 252
    cov = cluster_returns.cov() * 252

    sims = 10000
    weights = np.random.dirichlet(np.ones(len(mean_ret)), sims)
    res = np.array([stats(w, mean_ret, cov) for w in weights])
    best = res[:, 2].argmax()

    return (
        jsonify(
            {
                "closest_cluster": cluster,
                "optimized_companies": cluster_symbols,
                "optimal_portfolio": {t: round(w * 100, 2) for t, w in zip(mean_ret.index, weights[best])},
                "expected_return": float(res[best, 0]),
                "expected_volatility": float(res[best, 1]),
            }
        ),
        200,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
