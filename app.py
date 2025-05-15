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

import os, sys, requests, logging
import pandas as pd, numpy as np
from math import sqrt
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.cluster import AgglomerativeClustering

# ---------- 기본 설정 ----------
API_KEY = os.getenv("FMP_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 FMP_API_KEY 가 설정되지 않았습니다.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
RISK_FREE = 0.01
FEATURES = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

# ---------- Flask ----------
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

# ---------- 유틸 ----------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return float(100 - 100 / (1 + rs.iloc[-1]))

def fetch_company_data(symbol: str):
    prof  = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}'
    ratio = f'https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={API_KEY}'
    hist  = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={API_KEY}'
    try:
        p_json = requests.get(prof, timeout=10).json()[0]
        r_json = requests.get(ratio, timeout=10).json()[0]
        hist_j = requests.get(hist,  timeout=10).json()['historical']
    except Exception as e:
        logging.warning(f"{symbol}: API 호출 실패 → {e}")
        return None

    # 필수값 검사
    if p_json.get('pe') is None or p_json.get('beta') is None:
        logging.warning(f"{symbol}: 필수 재무지표 누락 → skip")
        return None

    hist_df = pd.DataFrame(hist_j).sort_values('date')
    closes  = hist_df['close']
    if len(closes) < 30:
        logging.warning(f"{symbol}: 가격 이력 부족 → skip")
        return None

    returns = closes.pct_change().dropna()
    return {
        'Ticker':         symbol,
        'PER':            p_json['pe'],
        'DividendYield':  r_json.get('dividendYieldPercentageTTM'),  # %
        'Beta':           p_json['beta'],
        'RSI':            calc_rsi(closes),
        'volume':         p_json.get('volAvg'),
        'Volatility':     returns.std() * sqrt(252),
        'returns':        returns
    }

# ---------- 데이터 수집 ----------
rows, returns_map = [], {}
for tkr in TICKERS:
    d = fetch_company_data(tkr)
    if d:
        rows.append({k: d[k] for k in ['Ticker'] + FEATURES})
        returns_map[tkr] = d['returns']

logging.info(f"성공 {len(rows)} 종목 / 실패 {len(TICKERS)-len(rows)} 종목")

if not rows:
    logging.error("가져온 종목이 없습니다. API 키·티커 확인 후 다시 배포하세요.")
    sys.exit(1)

cleaned_df = pd.DataFrame(rows)
cleaned_df_filtered = cleaned_df[FEATURES]

# ---------- 클러스터링 ----------
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cleaned_df['Cluster'] = agg.fit_predict(cleaned_df_filtered).astype('int')
centroids = cleaned_df.groupby('Cluster')[FEATURES].mean().reset_index(drop=True)

def map_input(cluster_feature, tag):
    return {'high': cluster_feature.max(),
            'medium': cluster_feature.mean(),
            'low': cluster_feature.min()}.get(tag, cluster_feature.mean())

def best_cluster(user_vals):
    dists = ((centroids - pd.Series(user_vals)).pow(2).sum(axis=1))
    return int(dists.idxmin())

def port_stats(w, mean_ret, cov):
    r = np.dot(w, mean_ret)
    v = sqrt(np.dot(w.T, np.dot(cov, w)))
    s = (r - RISK_FREE) / v
    return r, v, s

# ---------- Flask 엔드포인트 ----------
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    req = request.json
    if any(f not in req for f in FEATURES):
        return jsonify({'error': 'Missing required features'}), 400
    if any(req[f].lower() not in ['high', 'medium', 'low'] for f in FEATURES):
        return jsonify({'error': 'Values must be high / medium / low'}), 400

    user_vals = {f: map_input(centroids[f], req[f].lower()) for f in FEATURES}
    cl        = best_cluster(user_vals)
    tickers   = cleaned_df[cleaned_df['Cluster'] == cl]['Ticker'].tolist()

    rets = {t: returns_map[t] for t in tickers if t in returns_map}
    if not rets:
        return jsonify({'error': 'No return data for cluster'}), 400

    df   = pd.DataFrame(rets).dropna()
    mean = df.mean()*252
    cov  = df.cov()*252

    sims = 10000
    wts  = np.random.dirichlet(np.ones(len(mean)), sims)
    stats = np.array([port_stats(w, mean, cov) for w in wts])
    best = stats[:,2].argmax()

    return jsonify({
        'closest_cluster': cl,
        'optimized_companies': tickers,
        'optimal_portfolio': {t: round(w*100,2) for t,w in zip(mean.index, wts[best])},
        'expected_return': float(stats[best,0]),
        'expected_volatility': float(stats[best,1])
    }), 200

# ---------- 실행 ----------
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
