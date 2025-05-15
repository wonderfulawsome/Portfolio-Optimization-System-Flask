import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.cluster import KMeans
from scipy.optimize import minimize

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Flask 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

def fetch_company_data(symbol):
    profile_url = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}'
    ratios_url = f'https://financialmodelingprep.com/api/v3/ratios/{symbol}?limit=1&apikey={API_KEY}'
    historical_url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={API_KEY}'

    profile_resp = requests.get(profile_url)
    ratios_resp = requests.get(ratios_url)
    historical_resp = requests.get(historical_url)

    if profile_resp.status_code == 200 and ratios_resp.status_code == 200 and historical_resp.status_code == 200:
        profile_data = profile_resp.json()
        ratios_data = ratios_resp.json()
        historical_data = historical_resp.json()

        if profile_data and ratios_data and 'historical' in historical_data:
            profile = profile_data[0]
            ratios = ratios_data[0]
            historical = historical_data['historical']

            # 필요한 데이터 추출
            per = profile.get('pe', None)
            dividend_yield = profile.get('lastDiv', None)
            beta = profile.get('beta', None)
            volume = profile.get('volAvg', None)
            volatility = profile.get('volatility', None)
            rsi = None  # RSI는 별도의 계산이 필요합니다

            # 수익률 계산
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)
            df['return'] = df['close'].pct_change()
            returns = df['return'].dropna()

            return {
                'Ticker': symbol,
                'PER': per,
                'DividendYield': dividend_yield,
                'Beta': beta,
                'RSI': rsi,
                'volume': volume,
                'Volatility': volatility,
                'returns': returns
            }
    return None

# 예시 티커 리스트
tickers = ['AAPL', 'MSFT', 'GOOGL']  # 실제 사용하고자 하는 티커로 대체하세요

company_data_list = []
returns_data = {}

for ticker in tickers:
    data = fetch_company_data(ticker)
    if data:
        company_data_list.append({
            'Ticker': data['Ticker'],
            'PER': data['PER'],
            'DividendYield': data['DividendYield'],
            'Beta': data['Beta'],
            'RSI': data['RSI'],
            'volume': data['volume'],
            'Volatility': data['Volatility']
        })
        returns_data[ticker] = data['returns']

# 데이터프레임 생성
cleaned_df = pd.DataFrame(company_data_list).dropna()

features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']
cleaned_df_filtered = cleaned_df[features].dropna()

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(cleaned_df_filtered)
cleaned_df.loc[cleaned_df_filtered.index, 'Cluster'] = labels
cleaned_df['Cluster'] = cleaned_df['Cluster'].astype('Int64')

centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

def get_ret_vol_sr(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    required_features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "Missing required features"}), 400
    for feature in required_features:
        if data[feature].lower() not in ['high', 'medium', 'low']:
            return jsonify({"error": f"Invalid value for {feature}. Must be 'high', 'medium', or 'low'."}), 400

    user_input = {feature: data[feature].lower() for feature in required_features}
    mapped_values = {feature: map_input(centroids[feature], value) for feature, value in user_input.items()}
    closest_cluster = evaluate_cluster_fit(mapped_values, centroids)
    tickers_in_cluster = cleaned_df[cleaned_df['Cluster'] == closest_cluster]['Ticker'].tolist()

    cluster_returns = {ticker: returns_data[ticker] for ticker in tickers_in_cluster if ticker in returns_data}
    if not cluster_returns:
        return jsonify({"error": "No matching tickers found in stock data for the chosen cluster."}), 400

    returns_df = pd.DataFrame(cluster_returns).dropna()
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, len(mean_returns)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)

    for ind in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        all_weights[ind, :] = weights
        ret_vol_sr = get_ret_vol_sr(weights, mean_returns, cov_matrix)
        ret_arr[ind] = ret_vol_sr[0]
        vol_arr[ind] = ret_vol_sr[1]
        sharpe_arr[ind] = ret_vol_sr[2]

    max_sharpe_idx = sharpe_arr.argmax()
    optimal_weights = all_weights[max_sharpe_idx]
    optimal_portfolio = {ticker: round(weight * 100, 2) for ticker, weight in zip(mean_returns.index, optimal_weights)}

    result = {
        "closest_cluster": closest_cluster,
        "optimized_companies": tickers_in_cluster,
        "optimal_portfolio": optimal_portfolio,
        "expected_return": ret_arr[max_sharpe_idx],
        "expected_volatility": vol_arr[max_sharpe_idx]
    }
    return jsonify(result), 200
