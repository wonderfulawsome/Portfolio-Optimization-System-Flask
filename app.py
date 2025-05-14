import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
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

# CSV 파일 로드
cleaned_df = pd.read_csv("cleaned_data.csv")
stock_data = pd.read_csv("stock_data.csv")

features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']
cleaned_df_filtered = cleaned_df[features].dropna()

# KMeans 클러스터링
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(cleaned_df_filtered)
cleaned_df.loc[cleaned_df_filtered.index, 'Cluster'] = labels
cleaned_df['Cluster'] = cleaned_df['Cluster'].astype('Int64')

# 클러스터 중심 계산
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)

def map_input(cluster_feature, feature_value):
    if feature_value == 'high':
        return cluster_feature.max()
    elif feature_value == 'medium':
        return cluster_feature.mean()
    elif feature_value == 'low':
        return cluster_feature.min()
    else:
        return cluster_feature.mean()

def evaluate_cluster_fit(user_values, centroids):
    distances = []
    for idx, row in centroids.iterrows():
        distance = sum((user_values[feature] - row[feature])**2 for feature in user_values.keys())
        distances.append(distance)
    return distances.index(min(distances))

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

def get_ret_vol_sr(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

    stock_data_clean = stock_data.dropna(axis=1)
    ticker_columns = [col for col in stock_data_clean.columns if col != 'Date']
    cluster_tickers = list(set(tickers_in_cluster) & set(ticker_columns))
    if not cluster_tickers:
        return jsonify({"error": "No matching tickers found in stock data for the chosen cluster."}), 400

    stock_data_cluster = stock_data_clean[['Date'] + cluster_tickers]
    returns = stock_data_cluster.drop(columns=['Date']).pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

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
    optimal_portfolio = {ticker: round(weight * 100, 2) for ticker, weight in zip(cluster_tickers, optimal_weights)}

    result = {
        "closest_cluster": closest_cluster,
        "optimized_companies": tickers_in_cluster,
        "optimal_portfolio": optimal_portfolio,
        "expected_return": ret_arr[max_sharpe_idx],
        "expected_volatility": vol_arr[max_sharpe_idx]
    }
    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
