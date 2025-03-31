from flask import Flask, request, jsonify, send_file
from prophet import Prophet
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)
API_KEY = 'NE5PBQ8LFYJNMMGJ'  # 실제 키 입력

def get_cached_data(ticker):
    cache_file = f"cache_{ticker}.csv"

    # 캐시 파일이 존재하고, 생성된 지 24시간 안 됐는지 확인
    if os.path.exists(cache_file):
        modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.datetime.now() - modified_time < datetime.timedelta(hours=24):
            # 아직 유효하면 캐시 사용
            return pd.read_csv(cache_file, parse_dates=['date'])

    # (유효하지 않거나 캐시 없음) → Alpha Vantage API에서 새로 불러오기
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, _ = ts.get_daily(symbol=ticker, outputsize='full')
    data = data[['4. close']].rename(columns={'4. close': 'price'})
    data.index = pd.to_datetime(data.index)

    # 최근 5년치로 필터링
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=365 * 5)
    df = data[data.index >= cutoff_date].reset_index()
    df = df.rename(columns={"index": "date"})  # 'index' 열이 날짜 정보

    # 캐시 저장
    df.to_csv(cache_file, index=False)
    return df

def get_forecast_plot(ticker):
    # 캐시/불러오기
    df = get_cached_data(ticker)
    # Prophet 형식에 맞게 변환
    df_prophet = df.rename(columns={"date": "ds", "price": "y"})

    # Prophet 모델 훈련
    model = Prophet()
    model.fit(df_prophet)

    # 향후 7영업일 예측
    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)

    # 시각화
    fig = model.plot(forecast)
    plt.title(f"{ticker} 7-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()

    # 이미지로 반환
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return buf

@app.route('/forecast', methods=['POST'])
def forecast():
    data = request.get_json()
    ticker = data.get("ticker", "QQQ")
    try:
        image_buf = get_forecast_plot(ticker)
        return send_file(image_buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # QQQ.py 실행 시, Flask 서버 시작
    app.run(debug=True)
