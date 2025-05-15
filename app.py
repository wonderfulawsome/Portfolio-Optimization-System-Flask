TICKERS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", 
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", 
    "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", 
    "AIG", "AMT", "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", 
    "APA", "AAPL", "AMAT", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", 
    "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", 
    "BK", "BBWI", "BAX", "BDX", "BRK-B", "BBY", "BIO", "TECH", "BIIB", "BLK", 
    "BX", "BA", "BKNG", "BWA", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", 
    "BLDR", "BG", "BXP", "CHRW", "CDNS", "CZR", "CPT", "CPB", "COF", "CAH", 
    "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "COR", 
    "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", 
    "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME", "CMS", "KO", 
    "CTSH", "CL", "CMCSA", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", 
    "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD", "CCI", "CSX", 
    "CMI", "CVS", "DHR", "DRI", "DVA", "DAY", "DECK", "DE", "DAL", "DVN", 
    "DXCM", "FANG", "DLR", "DFS", "DG", "DLTR", "D", "DPZ", "DOV", "DOW", 
    "DHI", "DTE", "DUK", "DD", "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", 
    "EA", "ELV", "EMR", "ENPH", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", 
    "EQR", "ESS", "EL", "ETSY", "EG", "EVRG", "ES", "EXC", "EXPE", "EXPD", 
    "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB", 
    "FSLR", "FE", "FI", "FMC", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", 
    "FCX", "GRMN", "GE", "GEN", "GIS", "GM", "GPC", "GILD", "GPN", "GL", 
    "GS", "HAL", "HIG", "HAS", "HCA", "PEAK", "HSIC", "HSY", "HES", "HPE", 
    "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUM", "HBAN", 
    "HII", "IBM", "IEX", "IDXX", "ITW", "ILMN", "INCY", "IR", "PODD", "INTC", 
    "ICE", "IFF", "IP", "IPG", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", 
    "JBHT", "JKHY", "J", "JNJ", "JCI", "JPM", "JNPR", "K", "KDP", "KEY", 
    "KEYS", "KMB", "KIM", "KMI", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX", 
    "LW", "LVS", "LDOS", "LEN", "LNC", "LIN", "LYV", "LKQ", "LMT", "L", 
    "LOW", "LYB", "MTB", "MRO", "MPC", "MKTX", "MAR", "MMC", "MLM", "MAS", 
    "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD", 
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ", 
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX", 
    "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC", 
    "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC", 
    "ON", "OKE", "ORCL", "OGN", "OTIS", "PCAR", "PKG", "PANW", "PARA", "PH", 
    "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW", 
    "PXD", "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", 
    "PEG", "PTC", "PSA", "PHM", "QRVO", "PWR", "QCOM", "DGX", "RL", "RJF", 
    "RTX", "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "RHI", "ROK", 
    "ROL", "ROP", "ROST", "RCL", "SPGI", "CRM", "SBAC", "SLB", "STX", "SEE", 
    "SRE", "NOW", "SHW", "SPG", "SWKS", "SJM", "SNA", "SEDG", "SO", "LUV", 
    "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SYF", "SNPS", "SYY", "TMUS", 
    "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TFX", "TER", "TSLA", 
    "TXN", "TXT", "TMO", "TJX", "TSCO", "TT", "TDG", "TRV", "TRMB", "TFC", 
    "TYL", "TSN", "USB", "UDR", "ULTA", "UNP", "UAL", "UPS", "URI", "UNH", 
    "UHS", "VLO", "VTR", "VRSN", "VRSK", "VZ", "VRTX", "VFC", "VTRS", "VICI", 
    "V", "VMC", "WAB", "WBA", "WMT", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", 
    "WST", "WDC", "WRK", "WY", "WHR", "WMB", "WTW", "GWW", "WYNN", "XEL", 
    "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS"
]

import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.cluster import AgglomerativeClustering
from math import sqrt

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

API_KEY = os.environ.get("FMP_API_KEY")
RISK_FREE = 0.01
FEATURES = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

def calc_rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs.iloc[-1])

def fetch_company_data(symbol):
    prof = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}'
    hist = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={API_KEY}'
    p, h = requests.get(prof), requests.get(hist)
    if p.status_code != 200 or h.status_code != 200:
        return None
    p_json = p.json()[0]
    hist_df = pd.DataFrame(h.json()['historical']).sort_values('date')
    closes = hist_df['close']
    returns = closes.pct_change().dropna()
    return {
        'Ticker': symbol,
        'PER': p_json.get('pe'),
        'DividendYield': p_json.get('lastDiv'),
        'Beta': p_json.get('beta'),
        'RSI': calc_rsi(closes),
        'volume': p_json.get('volAvg'),
        'Volatility': returns.std() * sqrt(252),
        'returns': returns
    }


company_rows, returns_data = [], {}
for t in TICKERS:
    d = fetch_company_data(t)
    if d:
        company_rows.append({k: d[k] for k in ['Ticker'] + FEATURES})
        returns_data[t] = d['returns']

cleaned_df = pd.DataFrame(company_rows).dropna()
cleaned_df_filtered = cleaned_df[FEATURES]

# Agglomerative Clustering with 3 clusters
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = agg.fit_predict(cleaned_df_filtered)
cleaned_df['Cluster'] = labels.astype('Int64')

# centroids by mean of each cluster
centroids = cleaned_df.groupby('Cluster')[FEATURES].mean().reset_index(drop=True)

def map_input(cluster_feature, tag):
    return {'high': cluster_feature.max(),
            'medium': cluster_feature.mean(),
            'low': cluster_feature.min()}.get(tag, cluster_feature.mean())

def evaluate_cluster_fit(user_vals):
    dists = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dists.idxmin())

def portfolio_stats(w, mean_ret, cov):
    r = np.dot(w, mean_ret)
    v = sqrt(np.dot(w.T, np.dot(cov, w)))
    s = (r - RISK_FREE) / v
    return r, v, s

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

    user_vals = {f: map_input(centroids[f], data[f].lower()) for f in FEATURES}
    cluster = evaluate_cluster_fit(user_vals)
    tickers = cleaned_df[cleaned_df['Cluster'] == cluster]['Ticker'].tolist()

    rets = {t: returns_data[t] for t in tickers if t in returns_data}
    if not rets:
        return jsonify({'error': 'No return data for cluster'}), 400

    df = pd.DataFrame(rets).dropna()
    mean_ret, cov = df.mean() * 252, df.cov() * 252

    n, sims = len(mean_ret), 10000
    all_w = np.random.dirichlet(np.ones(n), sims)
    stats = np.array([portfolio_stats(w, mean_ret, cov) for w in all_w])
    best = stats[:, 2].argmax()

    result = {
        'closest_cluster': cluster,
        'optimized_companies': tickers,
        'optimal_portfolio': {t: round(w * 100, 2) for t, w in zip(mean_ret.index, all_w[best])},
        'expected_return': float(stats[best, 0]),
        'expected_volatility': float(stats[best, 1])
    }
    return jsonify(result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
