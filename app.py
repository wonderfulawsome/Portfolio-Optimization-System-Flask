import os
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Flask 설정
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 데이터 모델 (예시)
class Data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# CSV 파일 로드
cleaned_df = pd.read_csv("cleaned_data.csv")
# 기존 selected_data 대신 주가 데이터 파일 사용 (각 열에 종목, 행에 날짜별 주가)
stock_data = pd.read_csv("stock_data.csv")
# 주가 데이터가 없는 칼럼은 모두 삭제 (모든 값이 NaN인 열 제거)
stock_data = stock_data.dropna(axis=1, how='all')

# 클러스터링에 사용할 6개 특징 (HTML 및 API 입력과 일치)
features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

# NaN이 없는 행만 사용하여 클러스터링 수행
cleaned_df_filtered = cleaned_df[features].dropna()

# Spectral Clustering 수행
spectral = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10, random_state=42)
labels = spectral.fit_predict(cleaned_df_filtered)

# 클러스터링 결과를 원본 데이터프레임에 할당 (인덱스 일치)
cleaned_df.loc[cleaned_df_filtered.index, 'Cluster'] = labels
cleaned_df['Cluster'] = cleaned_df['Cluster'].astype('Int64')  # 정수형으로 변환

# 하드코딩된 클러스터 중심 (6개 변수)
centroids = pd.DataFrame({
    'PER': [28.629549, 24.969939, 22.338191, 22.468863],
    'DividendYield': [1.530, 1.805, 2.050, 1.455],
    'Beta': [1.0260, 1.0565, 1.0755, 0.9680],
    'RSI': [59.754312, 60.255378, 60.046894, 59.025935],
    'volume': [9412568, 3270264, 660531.5, 1488842],
    'Volatility': [0.014870, 0.015702, 0.016852, 0.016296]
}, index=[0, 1, 2, 3])

# 입력값에 따라 클러스터 특징값 매핑 함수
def map_input(cluster_feature, feature_value):
    if feature_value == 'high':
        return cluster_feature.max()
    elif feature_value == 'medium':
        return cluster_feature.mean()
    elif feature_value == 'low':
        return cluster_feature.min()
    else:
        return cluster_feature.mean()

# 사용자 입력과 클러스터 중심 사이의 유클리드 제곱거리 계산 함수
def evaluate_cluster_fit(user_values, centroids):
    distances = []
    for idx, row in centroids.iterrows():
        distance = sum((user_values[feature] - row[feature])**2 for feature in user_values.keys())
        distances.append(distance)
    return distances.index(min(distances))

# 포트폴리오 최적화를 위한 함수들
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio  # 최대화하기 위해 음수 반환

def get_ret_vol_sr(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

# 기본 페이지 제공 (index.html)
@app.route('/')
def index():
    return app.send_static_file('index.html')

# /optimize 엔드포인트: POST 요청으로 사용자 입력 받아 포트폴리오 최적화 수행
@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    required_features = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']
    
    # 입력 데이터 유효성 검사
    if not all(feature in data for feature in required_features):
        return jsonify({"error": "Missing required features"}), 400
    for feature in required_features:
        if data[feature].lower() not in ['high', 'medium', 'low']:
            return jsonify({"error": f"Invalid value for {feature}. Must be 'high', 'medium', or 'low'."}), 400

    # 전달받은 값을 소문자로 변환 후 매핑
    user_input = {feature: data[feature].lower() for feature in required_features}
    mapped_values = {feature: map_input(centroids[feature], value) for feature, value in user_input.items()}

    # 가장 적합한 클러스터 결정
    closest_cluster = evaluate_cluster_fit(mapped_values, centroids)

    # 해당 클러스터에 속한 전체 종목 추출
    tickers_in_cluster = cleaned_df[cleaned_df['Cluster'] == closest_cluster]['Ticker'].tolist()

    # stock_data의 컬럼(티커)과의 교집합 계산 (Date 열 제외)
    ticker_columns = [col for col in stock_data.columns if col != 'Date']
    cluster_tickers = list(set(tickers_in_cluster) & set(ticker_columns))
    if not cluster_tickers:
        return jsonify({"error": "No matching tickers found in stock data for the chosen cluster."}), 400

    # 최적화를 위해 stock_data에서 해당 종목들만 선택
    stock_data_cluster = stock_data[['Date'] + cluster_tickers]
    # (혹시라도) 주가 데이터가 없는 칼럼 삭제
    stock_data_cluster = stock_data_cluster.dropna(axis=1, how='all')
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

    # 포트폴리오 비중을 %로 표시 (가중치 * 100)
    portfolio_percentages = {
        ticker: float(optimal_weight * 100)
        for ticker, optimal_weight in zip(cluster_tickers, optimal_weights)
    }

    result = {
        "closest_cluster": closest_cluster,
        "optimized_companies": tickers_in_cluster,  # 전체 클러스터 종목 리스트
        "portfolio_percentages": portfolio_percentages,
        "expected_return": ret_arr[max_sharpe_idx],
        "expected_volatility": vol_arr[max_sharpe_idx]
    }

    return jsonify(result), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
