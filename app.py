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

import os, sys, logging, requests
import pandas as pd, numpy as np
from math import sqrt
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from sklearn.cluster import AgglomerativeClustering

API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise RuntimeError('API_KEY not set')

RISK_FREE = 0.01
FEATURES = ['PER', 'DividendYield', 'Beta', 'RSI', 'volume', 'Volatility']

# 로그
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

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

def calc_rsi(close, p=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(p).mean()
    loss = (-delta.clip(upper=0)).rolling(p).mean()
    rs = gain / loss
    return float(100 - 100 / (1 + rs.iloc[-1]))

def fetch(symbol):
    prof  = f'https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={API_KEY}'
    ratio = f'https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?apikey={API_KEY}'
    hist  = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&apikey={API_KEY}'
    try:
        p = requests.get(prof, timeout=10).json()[0]
        r = requests.get(ratio, timeout=10).json()[0]
        h = requests.get(hist, timeout=10).json()['historical']
    except Exception as e:
        logging.warning(f'{symbol}: API fail {e}')
        return None
    if p.get('pe') is None or p.get('beta') is None:
        return None
    df = pd.DataFrame(h).sort_values('date')
    if len(df) < 30:
        return None
    closes = df['close']
    returns = closes.pct_change().dropna()
    return {
        'Ticker': symbol,
        'PER': p['pe'],
        'DividendYield': r.get('dividendYieldPercentageTTM'),
        'Beta': p['beta'],
        'RSI': calc_rsi(closes),
        'volume': p.get('volAvg'),
        'Volatility': returns.std() * sqrt(252),
        'returns': returns
    }

rows, ret_map = [], {}
for t in TICKERS:
    d = fetch(t)
    if d: 
        rows.append({k: d[k] for k in ['Ticker'] + FEATURES})
        ret_map[t] = d['returns']
logging.info(f'Collected {len(rows)} / {len(TICKERS)} tickers')
if not rows:
    sys.exit('No data collected')

cleaned_df = pd.DataFrame(rows)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
cleaned_df['Cluster'] = agg.fit_predict(cleaned_df[FEATURES]).astype(int)
centroids = cleaned_df.groupby('Cluster')[FEATURES].mean().reset_index(drop=True)

def map_input(col, tag):
    return {'high': col.max(), 'medium': col.mean(), 'low': col.min()}.get(tag, col.mean())

def best_cluster(user):
    d = ((centroids - pd.Series(user)).pow(2).sum(axis=1))
    return int(d.idxmin())

def port_stats(w, mean, cov):
    r = np.dot(w, mean)
    v = sqrt(np.dot(w.T, cov @ w))
    s = (r - RISK_FREE) / v
    return r, v, s

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    req = request.json
    if any(f not in req for f in FEATURES):
        return jsonify({'error': 'missing feature'}), 400
    if any(req[f].lower() not in ['high','medium','low'] for f in FEATURES):
        return jsonify({'error': 'value must be high/medium/low'}), 400

    mapped = {f: map_input(centroids[f], req[f].lower()) for f in FEATURES}
    cl = best_cluster(mapped)
    tickers = cleaned_df[cleaned_df.Cluster == cl].Ticker.tolist()

    rets = {t: ret_map[t] for t in tickers if t in ret_map}
    if not rets:
        return jsonify({'error': 'no returns'}), 400
    df = pd.DataFrame(rets).dropna()
    mean, cov = df.mean()*252, df.cov()*252

    sims = 10000
    wts = np.random.dirichlet(np.ones(len(mean)), sims)
    stats = np.array([port_stats(w, mean, cov) for w in wts])
    best = stats[:,2].argmax()

    return jsonify({
        'closest_cluster': cl,
        'optimized_companies': tickers,
        'optimal_portfolio': {t: round(w*100,2) for t,w in zip(mean.index, wts[best])},
        'expected_return': float(stats[best,0]),
        'expected_volatility': float(stats[best,1])
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

