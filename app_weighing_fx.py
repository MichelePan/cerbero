import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

TTL_30_MIN = 60 * 30

@st.cache_data(ttl=TTL_30_MIN, show_spinner=False)
def download_fx_close_cached(tickers: tuple, start_date: datetime):
 df = yf.download(list(tickers), start=start_date, interval="1d", auto_adjust=True, progress=False)
 if df is None or df.empty:
 return None
 if "Close" not in df.columns:
 return None
 return df["Close"]

def render_weighing_fx():
 st.header("WEIGHING FX")
 st.markdown("---")

 with st.container(border=True):
 st.subheader("Impostazioni")

 months = st.slider(
 "Mesi Storico",
 min_value=3,
 max_value=24,
 value=12,
 step=1,
 help="Seleziona l'intervallo temporale in mesi (da 3 a 24)",
 key="wf_months",
 )

 st.markdown("### Selezione Tassi di Cambio")
 st.info("Seleziona fino a 6 coppie di valute.")

 DEFAULT_FX = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X"]

 common_pairs = [
 "EURUSD=X",
 "GBPUSD=X",
 "USDJPY=X",
 "USDCHF=X",
 "AUDUSD=X",
 "USDCAD=X",
 "NZDUSD=X",
 "EURGBP=X",
 "EURJPY=X",
 "GBPJPY=X",
 "USDMXN=X",
 "USDTRY=X",
 ]

 selected_fx = st.multiselect(
 "Scegli le valute (max 6):",
 options=common_pairs,
 default=DEFAULT_FX,
 max_selections=6,
 key="wf_selected_fx",
 )

 if not selected_fx:
 st.warning("Seleziona almeno un tasso di cambio per procedere.")
 st.stop()

 run = st.button("Calcola", type="primary", key="wf_run")

 if not run:
 st.info("Configura i parametri e premi **Calcola**.")
 return

 st.markdown("---")
 st.subheader(f"Analisi FX: Ultimi {months} Mesi")
 st.markdown("Dati basati su **valori di chiusura giornalieri** (1d).")

 start_date = datetime.now() - timedelta(days=months * 30.44)

 with st.spinner("Scarico dati (cached 30 min) e calcolo statistiche..."):
 try:
 data = download_fx_close_cached(tuple(selected_fx), start_date)
 if data is None or getattr(data, "empty", False):
 st.warning("Nessun dato trovato per le coppie di valute selezionate.")
 return

 percent_changes = data.pct_change().dropna()

 results_data = []
 for ticker in selected_fx:
 if ticker not in percent_changes.columns:
 continue

 changes = percent_changes[ticker]

 avg_change = changes.mean()

 neg_changes = changes[changes < 0]
 avg_neg = neg_changes.mean() if len(neg_changes) > 0 else 0.0

 pos_changes = changes[changes > 0]
 avg_pos = pos_changes.mean() if len(pos_changes) > 0 else 0.0

 abs_avg_neg = abs(avg_neg)
 abs_avg_pos = abs(avg_pos)

 if abs_avg_neg > abs_avg_pos:
 modified_pos = -abs_avg_pos
 final_val = avg_neg + modified_pos
 else:
 modified_neg = abs_avg_neg
 final_val = modified_neg + avg_pos

 results_data.append(
 {
 "Ticker": ticker,
 "Variazione Media": avg_change,
 "Media Negative": avg_neg,
 "Media Positive": avg_pos,
 "Somma Calcolata": final_val,
 }
 )

 df_results = pd.DataFrame(results_data)

 if df_results.empty:
 st.warning("Nessun dato trovato per le coppie di valute selezionate.")
 return

 df_results.set_index("Ticker", inplace=True)

 def color_cells(val):
 if val > 0:
 return "color: green; font-weight: bold"
 if val < 0:
 return "color: red; font-weight: bold"
 return "color: black"

 styled_df = df_results.style.applymap(color_cells, subset=["Variazione Media", "Media Negative", "Media Positive"])
 styled_df = styled_df.applymap(lambda x: "color: #0047AB; font-weight: bold", subset=["Somma Calcolata"])
 styled_df = styled_df.format(
 {
 "Variazione Media": "{:.2%}",
 "Media Negative": "{:.2%}",
 "Media Positive": "{:.2%}",
 "Somma Calcolata": "{:.2%}",
 }
 )

 st.dataframe(styled_df, use_container_width=True)

 except Exception as e:
 st.error(f"Si è verificato un errore: {e}")

 st.markdown("---")
 st.caption("Programma WEIGHING FX - Dati forniti da Yahoo Finance")
