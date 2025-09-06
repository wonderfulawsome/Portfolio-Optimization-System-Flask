import os
from math import sqrt
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_folder="static", static_url_path="")

# CORS 설정 - 모든 origin 허용
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

API_KEY = os.environ.get('API_KEY')
TICKERS = "AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX,JPM,JNJ"

def load_dummy_stock_data():
    """API_KEY가 없을 때 사용할 더미 데이터"""
    np.random.seed(42)
    tickers = TICKERS.split(',')
    
    data = []
    for i, ticker in enumerate(tickers):
        data.append({
            'Ticker': ticker,
            'pe': np.random.uniform(10, 40),
            'eps': np.random.uniform(1, 20),
            'marketCap': np.random.uniform(1e9, 2e12),
            'norm_price_diffs': np.random.normal(0, 1),
            'norm_price_ranges': np.random.normal(0, 1),
            'norm_volume_ratios': np.random.normal(0, 1)
        })
    
    return pd.DataFrame(data)

def load_stock_data():
    if API_KEY:
        try:
            url = f"https://financialmodelingprep.com/api/v3/quote/{TICKERS}?apikey={API_KEY}"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                price_diffs = [(stock.get('price', 100) - stock.get('priceAvg200', 100)) for stock in data]
                price_ranges = [(stock.get('yearHigh', 150) - stock.get('yearLow', 50)) for stock in data]
                volume_ratios = [(stock.get('volume', 1000000) / max(stock.get('avgVolume', 1000000), 1)) for stock in data]
                
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(np.column_stack([price_diffs, price_ranges, volume_ratios]))
                
                return pd.DataFrame([{
                    'Ticker': stock.get('symbol', f'TICKER{i}'),
                    'pe': stock.get('pe', np.random.uniform(10, 40)),
                    'eps': stock.get('eps', np.random.uniform(1, 20)),
                    'marketCap': stock.get('marketCap', np.random.uniform(1e9, 2e12)),
                    'norm_price_diffs': scaled_features[i, 0],
                    'norm_price_ranges': scaled_features[i, 1],
                    'norm_volume_ratios': scaled_features[i, 2]
                } for i, stock in enumerate(data)])
        except Exception as e:
            print(f"API Error: {e}")
    
    return load_dummy_stock_data()

def load_historical_data():
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252)
    tickers = TICKERS.split(',')
    
    data = {'Date': dates}
    for ticker in tickers:
        data[ticker] = np.random.normal(100, 10, 252).cumsum() + 100
    
    return pd.DataFrame(data)

# 초기화
cleaned_df = load_stock_data()
stock_data = load_historical_data()

features = ["pe", "eps", "marketCap", "norm_price_diffs", "norm_price_ranges", "norm_volume_ratios"]
cleaned_df_filtered = cleaned_df[features].dropna()

spectral = SpectralClustering(n_clusters=3, affinity='rbf', random_state=42)
labels = spectral.fit_predict(cleaned_df_filtered)
cleaned_df.loc[cleaned_df_filtered.index, "Cluster"] = labels.astype(int)

centroids = cleaned_df.groupby("Cluster")[features].mean().reset_index(drop=True)

def map_input(cluster_feature, tag):
    tags = {"high": cluster_feature.max(), "medium": cluster_feature.mean(), "low": cluster_feature.min()}
    return tags.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    dist = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dist.idxmin())

def stats(weights, mean_ret, cov):
    ret = np.dot(weights, mean_ret)
    vol = sqrt(np.dot(weights.T, np.dot(cov, weights)))
    sr = (ret - 0.01) / vol if vol > 0 else 0
    return ret, vol, sr

@app.after_request
def after_request(response):
    """모든 응답에 CORS 헤더 추가"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route("/")
def index():
    return jsonify({"status": "running", "api_key_configured": bool(API_KEY)})

@app.route("/optimize", methods=["POST", "OPTIONS"])
@cross_origin()
def optimize():
    # OPTIONS 요청 처리 (preflight)
    if request.method == "OPTIONS":
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # 입력 검증
        missing = [f for f in features if f not in data or not data[f]]
        if missing:
            return jsonify({"error": f"Missing features: {', '.join(missing)}"}), 400
        
        invalid = [f for f in features if data[f] and data[f].lower() not in ["high", "medium", "low"]]
        if invalid:
            return jsonify({"error": f"Invalid values for: {', '.join(invalid)}. Use high/medium/low"}), 400
        
        # 매핑 및 클러스터 찾기
        mapped = {f: map_input(centroids[f], data[f].lower()) for f in features}
        cluster = best_cluster(mapped)
        tickers = cleaned_df[cleaned_df["Cluster"] == cluster]["Ticker"].dropna().tolist()
        
        # 주식 데이터 필터링
        stock_data_clean = stock_data.dropna(axis=1)
        ticker_cols = [c for c in stock_data_clean.columns if c != "Date"]
        cluster_tickers = list(set(tickers) & set(ticker_cols))
        
        if not cluster_tickers:
            # 클러스터에 티커가 없으면 전체 티커 사용
            cluster_tickers = ticker_cols[:5]  # 최소 5개 티커 사용
        
        # 수익률 계산
        stock_cluster = stock_data_clean[["Date"] + cluster_tickers]
        returns = stock_cluster.drop(columns=["Date"]).pct_change().dropna()
        
        if returns.empty:
            return jsonify({"error": "Not enough data to calculate returns"}), 400
            
        mean_ret = returns.mean() * 252
        cov = returns.cov() * 252
        
        # 포트폴리오 최적화
        sims = 5000
        weights = np.random.dirichlet(np.ones(len(mean_ret)), sims)
        res = np.array([stats(w, mean_ret, cov) for w in weights])
        best = res[:, 2].argmax()
        
        response_data = {
            "closest_cluster": int(cluster),
            "optimized_companies": cluster_tickers,
            "optimal_portfolio": {t: round(w * 100, 2) for t, w in zip(mean_ret.index, weights[best])},
            "expected_return": float(res[best, 0]),
            "expected_volatility": float(res[best, 1]),
            "sharpe_ratio": float(res[best, 2])
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error in optimize: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "cors": "enabled"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
