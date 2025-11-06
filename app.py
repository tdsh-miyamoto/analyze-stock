#ライブラリの読み込み
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import yfinance as yf
import mplfinance as mpf
from statsmodels.tsa.arima.model import ARIMA

ticker = {"AMAZON": "AMZN", "APPLE": "AAPL"}
ticker_list = list(ticker.keys())
st.set_option('deprecation.showPyplotGlobalUse', False)

def pred_arima(data):
    train, test = data[:int(len(data)*0.7)], data[int(len(data)*0.7):]
    train = train["Close"].values
    test = test["Close"].values
    history = [x for x in train]
    model_pred = []
    for t in test:
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        model_pred.append(yhat)
        gt = t
        history.append(gt)
    return model_pred, test

#タイトル
st.title("株価分析アプリ")
st.write("個別カスタマイズ対応いたします")

# 以下をサイドバーに表示
st.sidebar.markdown("### 分析したい銘柄を選んでください")
#銘柄を指定
selected_ticker = st.sidebar.selectbox(
    '銘柄を選択：',
    ticker_list
)
ticker_code = ticker[selected_ticker]
df = yf.download(tickers = ticker_code, period = "6mo", interval = "1d", multi_level_index = False)
#print(df)
#データフレームを表示
st.markdown("### 入力データ")
st.dataframe(df.style.highlight_max(axis=0))
#mpfplotで可視化。
fig = mpf.plot(df, type="candle", volume=True, figratio=(10, 5), figsize= (12,8), style = "yahoo")
st.pyplot(fig)

#実行ボタン（なくてもよいが、その場合、処理を進めるまでエラー画面が表示されてしまう）
execute_pred = st.button("時系列予測を実行")
#実行ボタンを押したら下記を表示
if execute_pred:
    model_pred, test = pred_arima(df)
    #streamlit上で予測結果を表示
    fig = plt.figure(figsize= (12,8))
    plt.plot(test, color="Red", label="Ground Truth")
    plt.plot(model_pred, color="Blue", label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    st.pyplot(fig)

st.sidebar.markdown("### 分析したいデータのcsvファイルを指定してください")
#ファイルアップロード
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)
#ファイルがアップロードされたら以下が実行される
if uploaded_files:
    df = pd.read_csv(uploaded_files)
    #データフレームを表示
    st.markdown("### 入力データ")
    st.dataframe(df.style.highlight_max(axis=0))
    #mpfplotで可視化。
    fig = mpf.plot(df, type="candle", volume=True, figratio=(10, 5), figsize= (12,8))
    st.pyplot(fig)

    #実行ボタン（なくてもよいが、その場合、処理を進めるまでエラー画面が表示されてしまう）
    execute_pred = st.button("時系列予測を実行")
    #実行ボタンを押したら下記を表示
    if execute_pred:
        model_pred, test = pred_arima(df)
        #streamlit上で予測結果を表示
        fig = plt.figure(figsize= (12,8))
        plt.plot(test, color="Red", label="Ground Truth")
        plt.plot(model_pred, color="Blue", label="Prediction")
        plt.xlabel("Day")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        st.pyplot(fig)
