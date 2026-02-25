import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

TTL_30_MIN = 60 * 30

@st.cache_data(ttl=TTL_30_MIN, show_spinner=False)
def download_close_10y_cached(ticker: str):
 df = yf.download(ticker, period="10y", interval="1d", progress=False)
 if df is None or df.empty or "Close" not in df.columns:
 return None
 return df["Close"]

def render_fore_forecast_fx():
 st.header("FORE / FORECAST FX")
 st.markdown("---")

 with st.container(border=True):
 st.subheader("Impostazioni Analisi")

 ticker_map = {
 "EURUSD": "EURUSD=X",
 "GBPUSD": "GBPUSD=X",
 "USDCHF": "USDCHF=X",
 "USDCAD": "USDCAD=X",
 }

 c1, c2, c3 = st.columns([1, 1, 1])

 with c1:
 selected_pair_1 = st.selectbox("Tasso di Cambio 1", options=list(ticker_map.keys()), key="ff_pair1")
 selected_pair_2 = st.selectbox("Tasso di Cambio 2", options=list(ticker_map.keys()), key="ff_pair2")

 with c2:
 freq_options_map = {"Giornaliero": "D", "Settimanale": "W", "Mensile": "M"}
 freq_choice = st.selectbox("Frequenza Dati Storici", options=list(freq_options_map.keys()), key="ff_freq")
 freq_code = freq_options_map[freq_choice]

 if freq_choice == "Giornaliero":
 forecast_label = "Periodo di Previsione (Giorni)"
 forecast_options = [30, 60, 90]
 plot_freq_str = "Giorni"
 date_freq_code = "B"
 elif freq_choice == "Settimanale":
 forecast_label = "Periodo di Previsione (Settimane)"
 forecast_options = [4, 8, 12, 24]
 plot_freq_str = "Settimane"
 date_freq_code = "W-MON"
 else:
 forecast_label = "Periodo di Previsione (Mesi)"
 forecast_options = [1, 2, 3, 6]
 plot_freq_str = "Mesi"
 date_freq_code = "MS"

 forecast_period = st.selectbox(forecast_label, options=forecast_options, key="ff_forecast_period")

 with c3:
 lookback_options = {
 "Ultimi 60 valori": 60,
 "Ultimi 120 valori": 120,
 "Ultimi 240 valori": 240,
 "Ultimi 360 valori": 360,
 "Ultimi 720 valori": 720,
 }
 lookback_choice = st.selectbox("Periodo Storico (N valori)", options=list(lookback_options.keys()), key="ff_lookback")
 lookback_value = lookback_options[lookback_choice]

 model_options = {"ARMA(2, 0, 2)": (2, 0, 2), "ARIMA(2, 1, 2)": (2, 1, 2)}
 model_choice_name = st.selectbox("Modello Statistico", options=list(model_options.keys()), key="ff_model")
 model_order = model_options[model_choice_name]

 calculate_button = st.button("Calcola", type="primary", key="ff_calculate")

 ticker1 = ticker_map[selected_pair_1]
 ticker2 = ticker_map[selected_pair_2]

 def calculate_adf(series):
 try:
 result = adfuller(series.dropna())
 return result[0], result[1]
 except Exception:
 return None, None

 if not calculate_button:
 st.info("Configura i parametri e premi **Calcola**.")
 return

 if ticker1 == ticker2:
 st.error("Per favore, seleziona due tassi di cambio diversi.")
 return

 with st.spinner("Scaricamento dati (cached 30 min), elaborazione e calcolo modello..."):
 try:
 raw_data1 = download_close_10y_cached(ticker1)
 raw_data2 = download_close_10y_cached(ticker2)

 if raw_data1 is None or raw_data2 is None:
 st.error("Dati non disponibili per uno o entrambi i ticker selezionati.")
 return

 df1 = pd.DataFrame(raw_data1).reset_index()
 df2 = pd.DataFrame(raw_data2).reset_index()
 df1.columns = ["Data", "Valore"]
 df2.columns = ["Data", "Valore"]

 df1 = df1.sort_values("Data")
 df2 = df2.sort_values("Data")

 df1.set_index("Data", inplace=True)
 df2.set_index("Data", inplace=True)

 if freq_code == "W":
 df1 = df1.resample("W").last()
 df2 = df2.resample("W").last()
 elif freq_code == "M":
 df1 = df1.resample("M").last()
 df2 = df2.resample("M").last()

 df1 = df1.reset_index().dropna()
 df2 = df2.reset_index().dropna()

 df1 = df1.tail(lookback_value).reset_index(drop=True)
 df2 = df2.tail(lookback_value).reset_index(drop=True)

 model1_fit = ARIMA(df1["Valore"], order=model_order).fit()
 model2_fit = ARIMA(df2["Valore"], order=model_order).fit()

 forecast1 = model1_fit.forecast(steps=forecast_period)
 forecast2 = model2_fit.forecast(steps=forecast_period)

 last_date = df1["Data"].iloc[-1]
 forecast_dates = pd.date_range(start=last_date, periods=forecast_period + 1, freq=date_freq_code)[1:]

 preds1 = model1_fit.fittedvalues
 preds2 = model2_fit.fittedvalues

 common_idx1 = df1.index.intersection(preds1.index)
 rmse1 = np.sqrt(mean_squared_error(df1.loc[common_idx1, "Valore"], preds1.loc[common_idx1])) if len(common_idx1) > 0 else 0

 common_idx2 = df2.index.intersection(preds2.index)
 rmse2 = np.sqrt(mean_squared_error(df2.loc[common_idx2, "Valore"], preds2.loc[common_idx2])) if len(common_idx2) > 0 else 0

 adf_stat1, adf_p1 = calculate_adf(df1["Valore"])
 adf_stat2, adf_p2 = calculate_adf(df2["Valore"])

 min_len = min(len(df1), len(df2))
 coint_test = coint(df1["Valore"].iloc[-min_len:], df2["Valore"].iloc[-min_len:])

 fig = go.Figure()
 fig.add_trace(go.Scatter(x=df1["Data"], y=df1["Valore"], mode="lines", name=f"{selected_pair_1} - Storico", line=dict(color="blue")))
 fig.add_trace(go.Scatter(x=df2["Data"], y=df2["Valore"], mode="lines", name=f"{selected_pair_2} - Storico", line=dict(color="red")))
 fig.add_trace(go.Scatter(x=forecast_dates, y=forecast1, mode="lines", name=f"{selected_pair_1} - Previsione", line=dict(color="blue", dash="dash")))
 fig.add_trace(go.Scatter(x=forecast_dates, y=forecast2, mode="lines", name=f"{selected_pair_2} - Previsione", line=dict(color="red", dash="dash")))

 fig.update_layout(
 title=f"Modello {model_choice_name} ({freq_choice}) - Previsione a {forecast_period} {plot_freq_str}",
 xaxis_title="Data",
 yaxis_title="Valore",
 hovermode="x unified",
 )
 st.plotly_chart(fig, use_container_width=True)

 col1, col2 = st.columns(2)

 with col1:
 st.subheader(f"Statistiche: {selected_pair_1}", divider="blue")
 st.write(f"**Valore Medio:** {df1['Valore'].mean():.4f}")
 st.write(f"**Valore Minimo:** {df1['Valore'].min():.4f}")
 st.write(f"**Valore Massimo:** {df1['Valore'].max():.4f}")

 st.markdown("**Test Stazionarietà (ADF):**")
 if adf_stat1 is not None:
 st.write(f"Statistic: {adf_stat1:.4f}")
 st.write(f"P-value: {adf_p1:.4f} {'(Stazionario)' if adf_p1 < 0.05 else '(Non Stazionario)'}")

 st.markdown("**Metriche Modello:**")
 st.write(f"AIC: {model1_fit.aic:.4f}")
 st.write(f"BIC: {model1_fit.bic:.4f}")
 st.write(f"RMSE: {rmse1:.4f}")

 with col2:
 st.subheader(f"Statistiche: {selected_pair_2}", divider="red")
 st.write(f"**Valore Medio:** {df2['Valore'].mean():.4f}")
 st.write(f"**Valore Minimo:** {df2['Valore'].min():.4f}")
 st.write(f"**Valore Massimo:** {df2['Valore'].max():.4f}")

 st.markdown("**Test Stazionarietà (ADF):**")
 if adf_stat2 is not None:
 st.write(f"Statistic: {adf_stat2:.4f}")
 st.write(f"P-value: {adf_p2:.4f} {'(Stazionario)' if adf_p2 < 0.05 else '(Non Stazionario)'}")

 st.markdown("**Metriche Modello:**")
 st.write(f"AIC: {model2_fit.aic:.4f}")
 st.write(f"BIC: {model2_fit.bic:.4f}")
 st.write(f"RMSE: {rmse2:.4f}")

 st.subheader("Analisi di Cointegrazione", divider="gray")
 coint_col1, coint_col2 = st.columns([2, 1])
 with coint_col1:
 st.write(f"Test Statistic: {coint_test[0]:.4f}")
 st.write(f"P-value: {coint_test[1]:.4f}")
 st.write(
 f"Critical Values (1%): {coint_test[2][0]:.4f}, (5%): {coint_test[2][1]:.4f}, (10%): {coint_test[2][2]:.4f}"
 )
 with coint_col2:
 if coint_test[1] < 0.05:
 st.success("Risultato: Cointegrati")
 else:
 st.warning("Risultato: Non cointegrati")

 st.markdown("---")
 st.subheader(f"Previsioni Future ({plot_freq_str})")

 st.markdown(
 """
 <style>
 .forecast-box {
 background-color: #ffffcc;
 padding: 16px;
 border-radius: 6px;
 border: 1px solid #ddd;
 }
 </style>
 """,
 unsafe_allow_html=True,
 )

 fc_col1, fc_col2 = st.columns(2)
 with fc_col1:
 st.markdown(f"<div class='forecast-box'><h3 style='color:blue'>{selected_pair_1}</h3>", unsafe_allow_html=True)
 st.write(f"**1° Periodo:** {forecast1.iloc[0]:.4f}")
 st.write(f"**Ultimo Periodo:** {forecast1.iloc[-1]:.4f}")
 st.write(f"**Media previsione:** {forecast1.mean():.4f}")
 st.markdown("</div>", unsafe_allow_html=True)

 with fc_col2:
 st.markdown(f"<div class='forecast-box'><h3 style='color:red'>{selected_pair_2}</h3>", unsafe_allow_html=True)
 st.write(f"**1° Periodo:** {forecast2.iloc[0]:.4f}")
 st.write(f"**Ultimo Periodo:** {forecast2.iloc[-1]:.4f}")
 st.write(f"**Media previsione:** {forecast2.mean():.4f}")
 st.markdown("</div>", unsafe_allow_html=True)

 except Exception as e:
 st.error(f"Si è verificato un errore: {str(e)}")
 st.write("Controllare la connessione internet o i ticker selezionati.")
