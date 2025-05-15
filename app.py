import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.cluster import KMeans
from math import sqrt

# Flask 기본 설정
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

API_KEY = os.environ.get("API_KEY")          # 환경변수에 저장된 키
RISK_FREE = 0.01
FEATURES = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

# 간단한 RSI 계산
def calc_rsi(close, period: int = 14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# FMP 데이터 수집
def fetch_company_data(symbol):
    prof = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}'
    hist = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={API_KEY}'
    prof_r = requests.get(prof)
    hist_r = requests.get(hist)
    if prof_r.status_code != 200 or hist_r.status_code != 200:
        return None
    prof_json = prof_r.json()[0]
    hist_json = pd.DataFrame(hist_r.json()['historical'])
    hist_json['date'] = pd.to_datetime(hist_json['date'])
    hist_json.sort_values('date', inplace=True)
    closes = hist_json['close']
    returns = closes.pct_change().dropna()
    return {
        'Ticker': symbol,
        'PER': prof_json.get('pe'),
        'DividendYield': prof_json.get('lastDiv'),
        'Beta': prof_json.get('beta'),
        'RSI': calc_rsi(closes),
        'volume': prof_json.get('volAvg'),
        'Volatility': returns.std() * sqrt(252),
        'returns': returns
    }

# 분석 대상 티커
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA']  # 필요 시 수정
company_data, returns_data = [], {}
for t in TICKERS:
    res = fetch_company_data(t)
    if res:
        company_data.append({k: res[k] for k in ['Ticker'] + FEATURES})
        returns_data[t] = res['returns']

cleaned_df = pd.DataFrame(company_data).dropna()
cleaned_df_filtered = cleaned_df[FEATURES]

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(cleaned_df_filtered)
cleaned_df['Cluster'] = labels.astype('Int64')
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)

def map_input(cluster_feature, tag):
    if tag == 'high':
        return cluster_feature.max()
    if tag == 'low':
        return cluster_feature.min()
    return cluster_feature.mean()

def evaluate_cluster_fit(user_vals):
    dists = ((centroids[FEATURES] - pd.Series(user_vals)).pow(2).sum(axis=1))
    return dists.idxmin()

def portfolio_stats(weights, mean_returns, cov_matrix):
    port_ret = np.dot(weights, mean_returns)
    port_vol = sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_ret - RISK_FREE) / port_vol
    return port_ret, port_vol, sharpe

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    if any(f not in data for f in FEATURES):
        return jsonify({'error': 'Missing required features'}), 400
    if any(data[f].lower() not in ['high', 'medium', 'low'] for f in FEATURES):
        return jsonify({'error': 'Feature values must be high, medium, or low'}), 400

    user_map = {f: map_input(centroids[f], data[f].lower()) for f in FEATURES}
    cluster = evaluate_cluster_fit(user_map)
    cluster_tickers = cleaned_df[cleaned_df['Cluster'] == cluster]['Ticker'].tolist()

    rets = {t: returns_data[t] for t in cluster_tickers if t in returns_data}
    if not rets:
        return jsonify({'error': 'No return data for cluster'}), 400

    ret_df = pd.DataFrame(rets).dropna()
    mean_ret = ret_df.mean() * 252
    cov = ret_df.cov() * 252

    n = len(mean_ret)
    sims = 10000
    all_w, ret_arr, vol_arr, sharpe_arr = np.zeros((sims, n)), np.zeros(sims), np.zeros(sims), np.zeros(sims)
    for i in range(sims):
        w = np.random.rand(n)
        w /= w.sum()
        all_w[i] = w
        ret_arr[i], vol_arr[i], sharpe_arr[i] = portfolio_stats(w, mean_ret, cov)

    best = sharpe_arr.argmax()
    opt_weights = all_w[best]
    result = {
        'closest_cluster': int(cluster),
        'optimized_companies': cluster_tickers,
        'optimal_portfolio': {t: round(w * 100, 2) for t, w in zip(mean_ret.index, opt_weights)},
        'expected_return': float(ret_arr[best]),
        'expected_volatility': float(vol_arr[best])
    }
    return jsonify(result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
