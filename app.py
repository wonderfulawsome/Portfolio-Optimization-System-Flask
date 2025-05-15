import os
from math import sqrt

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sklearn.cluster import AgglomerativeClustering

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

# CSV 파일 로드
cleaned_df = pd.read_csv("cleaned_data.csv")
stock_data = pd.read_csv("stock_data.csv")

features = ["PER", "DividendYield", "Beta", "RSI", "volume", "Volatility"]
cleaned_df_filtered = cleaned_df[features].dropna()

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = agg.fit_predict(cleaned_df_filtered)
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

    tickers = cleaned_df[cleaned_df["Cluster"] == cluster]["Ticker"].dropna().tolist()
    stock_data_clean = stock_data.dropna(axis=1)
    ticker_cols = [c for c in stock_data_clean.columns if c != "Date"]
    cluster_tickers = list(set(tickers) & set(ticker_cols))
    if not cluster_tickers:
        return jsonify({"error": "No matching tickers in stock data"}), 400

    stock_cluster = stock_data_clean[["Date"] + cluster_tickers]
    returns = stock_cluster.drop(columns=["Date"]).pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

    sims = 10000
    weights = np.random.dirichlet(np.ones(len(mean_ret)), sims)
    res = np.array([stats(w, mean_ret, cov) for w in weights])
    best = res[:, 2].argmax()

    return (
        jsonify(
            {
                "closest_cluster": cluster,
                "optimized_companies": cluster_tickers,
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
