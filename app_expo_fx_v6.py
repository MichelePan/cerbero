import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta, date

def render_expo_fx_v6():
    PREFIX = "expov6__"
    TTL_30_MIN = 60 * 30

    def k(name: str) -> str:
        return f"{PREFIX}{name}"

    st.markdown(
        """
        <style>
        div[data-testid="stDateInput"] > div > div > input,
        div[data-testid="stSelectbox"] > div > div > select,
        div[data-testid="stTextInput"] > div > div > input,
        div[data-testid="stNumberInput"] > div > div > input {
          background-color: #FFFF00 !important;
          color: black;
          font-weight: bold;
        }

        .slot-container {
          border: 2px solid #555;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 25px;
          background-color: #ffffff;
          box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .slot-yellow { border: 3px solid #FFD700 !important; background-color: #fffff0; }
        .slot-blue { border: 3px solid #0047AB !important; background-color: #f0f8ff; }
        .slot-red { border: 3px solid #FF0000 !important; background-color: #fff0f0; }

        .slot-header {
          font-size: 18px;
          font-weight: bold;
          color: #333;
          margin-bottom: 10px;
          border-bottom: 1px solid #ddd;
          padding-bottom: 5px;
        }

        .trade-separator { border-top: 1px dashed #aaa; margin: 10px 0; }

        .value-positive { color: #0047AB !important; font-weight: bold; font-size: 18px; background-color: #e6f0ff; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #0047AB; margin-top: 5px; }
        .value-negative { color: #FF0000 !important; font-weight: bold; font-size: 18px; background-color: #ffe6e6; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #FF0000; margin-top: 5px; }
        .value-neutral  { color: #333; font-size: 18px; background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center; margin-top: 5px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_data(ttl=TTL_30_MIN, show_spinner=False)
    def download_data_cached(symbol: str, start_d: date, end_d: date):
        if not symbol:
            return None
        sym = symbol.strip().upper()
        if not sym.endswith("=X"):
            sym += "=X"
        try:
            df = yf.download(sym, start=start_d, end=end_d, progress=False, group_by="ticker", interval="1d")
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df = df[sym]
            if "Close" not in df.columns:
                return None
            return df
        except Exception:
            return None

    def download_data(symbol: str, start_d: date, end_d_dt: datetime):
        end_day = end_d_dt.date()
        return download_data_cached(symbol, start_d, end_day)

    def calculate_daily_pnl(df, open_p, type_t, pipv, swap_d):
        if df is None or df.empty or "Close" not in df.columns:
            return None
        closes = df["Close"]
        if closes.empty:
            return None

        pnl_series = []
        days_counter = 0
        for price in closes:
            days_counter += 1
            raw_diff = (open_p - price) if type_t == "SELL" else (price - open_p)
            pips = raw_diff * 10000
            pnl = (pips * pipv) + (swap_d * days_counter)
            pnl_series.append(pnl)

        return pd.Series(pnl_series, index=closes.index)

    def reset_all():
        for key in list(st.session_state.keys()):
            if key.startswith(PREFIX):
                del st.session_state[key]
        st.rerun()

    def make_opposite_callback(source_key, target_key):
        def callback():
            cur = st.session_state[source_key]
            st.session_state[target_key] = "SELL" if cur == "BUY" else "BUY"
        return callback

    def make_sync_callback(source_key, target_key):
        def callback():
            st.session_state[target_key] = st.session_state[source_key]
        return callback

    # ---- init state
    if k("inputs_initialized") not in st.session_state:
        default_date = date(2017, 1, 1)

        for i in range(1, 16):
            if 1 <= i <= 5:
                sym1, sym2 = "EURUSD", "USDCHF"
            elif 6 <= i <= 10:
                sym1, sym2 = "EURUSD", "GBPUSD"
            else:
                sym1, sym2 = "GBPUSD", "USDCAD"

            st.session_state[k(f"s{i}_t1_date")] = default_date
            st.session_state[k(f"s{i}_t1_type")] = "BUY"
            st.session_state[k(f"s{i}_t1_sym")] = sym1
            st.session_state[k(f"s{i}_t1_open")] = 0.0
            st.session_state[k(f"s{i}_t1_swap")] = 0.0
            st.session_state[k(f"s{i}_t1_pipv")] = 0.0

            st.session_state[k(f"s{i}_t2_date")] = default_date
            st.session_state[k(f"s{i}_t2_type")] = "SELL"
            st.session_state[k(f"s{i}_t2_sym")] = sym2
            st.session_state[k(f"s{i}_t2_open")] = 0.0
            st.session_state[k(f"s{i}_t2_swap")] = 0.0
            st.session_state[k(f"s{i}_t2_pipv")] = 0.0

        st.session_state[k("dispy")] = 0.0
        st.session_state[k("ctrv")] = 0.0
        st.session_state[k("results")] = {}
        st.session_state[k("inputs_initialized")] = True

    st.header("EXPO FX V6 - Dashboard Trading")

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("AGGIORNA DATI (UPDATE)", type="primary", key=k("btn_update")):
            st.session_state[k("do_update")] = True
    with c_btn2:
        if st.button("RESET TUTTO", type="secondary", key=k("btn_reset")):
            reset_all()

    c_d, c_c = st.columns(2)
    with c_d:
        st.number_input("DISPY", value=float(st.session_state.get(k("dispy"), 0.0)), format="%.2f", key=k("dispy"))
    with c_c:
        st.number_input("CTRV", value=float(st.session_state.get(k("ctrv"), 0.0)), format="%.2f", key=k("ctrv"))

    st.markdown("---")

    # ---- compute
    need_update = st.session_state.get(k("do_update"), False) or not st.session_state.get(k("results"))
    if need_update:
        if k("do_update") in st.session_state:
            del st.session_state[k("do_update")]

        SLOT_COLORS = [
            "#DC143C", "#801818", "#FFA500", "#FF0000", "#CD5700",
            "#99CBFF", "#1560BD", "#00FFFF", "#0000CD", "#5F9EA0",
            "#8FBC8F", "#228B22", "#66FF00", "#008000", "#00A86B",
        ]

        new_results = {}
        all_chart_traces = []
        group_series = {"A": [], "B": [], "C": []}

        with st.spinner("Aggiornamento dati FX (cache 30 minuti)..."):
            for i in range(1, 16):
                t1_date = st.session_state.get(k(f"s{i}_t1_date"))
                t1_type = st.session_state.get(k(f"s{i}_t1_type"))
                t1_sym = st.session_state.get(k(f"s{i}_t1_sym"), "")
                t1_open = float(st.session_state.get(k(f"s{i}_t1_open"), 0.0))
                t1_swap = float(st.session_state.get(k(f"s{i}_t1_swap"), 0.0))
                t1_pipv = float(st.session_state.get(k(f"s{i}_t1_pipv"), 0.0))

                t2_date = st.session_state.get(k(f"s{i}_t2_date"))
                t2_type = st.session_state.get(k(f"s{i}_t2_type"))
                t2_sym = st.session_state.get(k(f"s{i}_t2_sym"), "")
                t2_open = float(st.session_state.get(k(f"s{i}_t2_open"), 0.0))
                t2_swap = float(st.session_state.get(k(f"s{i}_t2_swap"), 0.0))
                t2_pipv = float(st.session_state.get(k(f"s{i}_t2_pipv"), 0.0))

                res = {
                    "t1_act": 0.0, "t1_val": 0.0, "t1_err": None,
                    "t2_act": 0.0, "t2_val": 0.0, "t2_err": None,
                    "net_series": None,
                }

                df1 = None
                if t1_sym and t1_open > 0:
                    df1 = download_data(t1_sym, t1_date, datetime.now())
                    if df1 is None:
                        res["t1_err"] = "No Data"
                    else:
                        s1 = calculate_daily_pnl(df1, t1_open, t1_type, t1_pipv, t1_swap)
                        if s1 is not None and not s1.empty:
                            res["t1_act"] = float(df1["Close"].iloc[-1])
                            res["t1_val"] = float(s1.iloc[-1])
                        else:
                            res["t1_err"] = "Err Calc"

                df2 = None
                if t2_sym and t2_open > 0:
                    df2 = download_data(t2_sym, t2_date, datetime.now())
                    if df2 is None:
                        res["t2_err"] = "No Data"
                    else:
                        s2 = calculate_daily_pnl(df2, t2_open, t2_type, t2_pipv, t2_swap)
                        if s2 is not None and not s2.empty:
                            res["t2_act"] = float(df2["Close"].iloc[-1])
                            res["t2_val"] = float(s2.iloc[-1])
                        else:
                            res["t2_err"] = "Err Calc"

                if df1 is not None or df2 is not None:
                    all_indices = []
                    if df1 is not None and not df1.empty:
                        all_indices.append(df1.index)
                    if df2 is not None and not df2.empty:
                        all_indices.append(df2.index)

                    if all_indices:
                        common_idx = pd.DatetimeIndex(sorted(set().union(*all_indices)))
                        net_pnl = pd.Series(0.0, index=common_idx)

                        if df1 is not None and not df1.empty:
                            pnl1 = calculate_daily_pnl(df1, t1_open, t1_type, t1_pipv, t1_swap)
                            if pnl1 is not None:
                                net_pnl += pnl1.reindex(common_idx).ffill().fillna(0)

                        if df2 is not None and not df2.empty:
                            pnl2 = calculate_daily_pnl(df2, t2_open, t2_type, t2_pipv, t2_swap)
                            if pnl2 is not None:
                                net_pnl += pnl2.reindex(common_idx).ffill().fillna(0)

                        res["net_series"] = pd.DataFrame({"Date": net_pnl.index, "Value": net_pnl.values})

                        all_chart_traces.append(
                            {"name": f"Slot {i}", "df": res["net_series"], "color": SLOT_COLORS[i - 1]}
                        )

                        s_group = pd.Series(net_pnl.values, index=net_pnl.index)
                        if 1 <= i <= 5:
                            group_series["A"].append(s_group)
                        elif 6 <= i <= 10:
                            group_series["B"].append(s_group)
                        else:
                            group_series["C"].append(s_group)

                new_results[i] = res

        st.session_state[k("results")] = new_results
        st.session_state[k("chart_data")] = all_chart_traces
        st.session_state[k("dispy_val")] = float(st.session_state.get(k("dispy"), 0.0))
        st.session_state[k("ctrv_val")] = float(st.session_state.get(k("ctrv"), 0.0))

        group_totals = {}
        for g_key, s_list in group_series.items():
            if not s_list:
                group_totals[g_key] = None
                continue
            all_idx = pd.DatetimeIndex(sorted(set().union(*[s.index for s in s_list])))
            total = pd.Series(0.0, index=all_idx)
            for s in s_list:
                total += s.reindex(all_idx).ffill().fillna(0)
            group_totals[g_key] = pd.DataFrame({"Date": total.index, "Value": total.values})

        st.session_state[k("group_totals")] = group_totals

    # ---- read state
    current_results = st.session_state.get(k("results"), {})
    current_chart_data = st.session_state.get(k("chart_data"), [])
    current_group_totals = st.session_state.get(k("group_totals"), {})
    current_dispy = float(st.session_state.get(k("dispy_val"), 0.0))
    current_ctrv = float(st.session_state.get(k("ctrv_val"), 0.0))

    if current_chart_data:
        all_dates = pd.concat([t["df"]["Date"] for t in current_chart_data])
        min_x = all_dates.min()
        max_x = all_dates.max()
    else:
        min_x = datetime.now() - timedelta(days=30)
        max_x = datetime.now()

    def display_colored_value(value, error_msg=None):
        if error_msg:
            st.markdown(f'<div class="value-neutral">{error_msg}</div>', unsafe_allow_html=True)
        else:
            if value > 0:
                st.markdown(f'<div class="value-positive">{value:.2f}</div>', unsafe_allow_html=True)
            elif value < 0:
                st.markdown(f'<div class="value-negative">{value:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="value-neutral">{value:.2f}</div>', unsafe_allow_html=True)

    def add_zero_line(fig, min_date, max_date):
        fig.add_shape(
            type="line",
            x0=min_date,
            y0=0,
            x1=max_date,
            y1=0,
            line=dict(color="black", width=2, dash="dash"),
        )

    # ---- UI slots
    for i in range(1, 16):
        data = current_results.get(
            i,
            {"t1_act": 0.0, "t1_val": 0.0, "t1_err": None, "t2_act": 0.0, "t2_val": 0.0, "t2_err": None},
        )

        if 1 <= i <= 5:
            border_class = "slot-container slot-yellow"
        elif 6 <= i <= 10:
            border_class = "slot-container slot-blue"
        else:
            border_class = "slot-container slot-red"

        st.markdown(f'<div class="{border_class}"><div class="slot-header">SLOT {i}</div></div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 0.8, 1, 0.8, 0.8, 0.8, 0.8, 1])
        with c1:
            st.date_input("DATE", key=k(f"s{i}_t1_date"), on_change=make_sync_callback(k(f"s{i}_t1_date"), k(f"s{i}_t2_date")))
        with c2:
            st.selectbox("TYPE", ["BUY", "SELL"], key=k(f"s{i}_t1_type"), on_change=make_opposite_callback(k(f"s{i}_t1_type"), k(f"s{i}_t2_type")))
        with c3:
            st.text_input("SYMBOL", key=k(f"s{i}_t1_sym"))
        with c4:
            st.number_input("OPEN", value=0.0, format="%.5f", key=k(f"s{i}_t1_open"))
        with c5:
            st.metric("ACTUAL", data["t1_err"] if data["t1_err"] else f"{data['t1_act']:.5f}")
        with c6:
            st.number_input("SWAP/D", value=0.0, format="%.2f", key=k(f"s{i}_t1_swap"))
        with c7:
            st.number_input("PIPV", value=0.0, format="%.2f", key=k(f"s{i}_t1_pipv"), on_change=make_sync_callback(k(f"s{i}_t1_pipv"), k(f"s{i}_t2_pipv")))
        with c8:
            display_colored_value(data["t1_val"], data["t1_err"])

        st.markdown('<div class="trade-separator"></div>', unsafe_allow_html=True)

        d1, d2, d3, d4, d5, d6, d7, d8 = st.columns([1, 0.8, 1, 0.8, 0.8, 0.8, 0.8, 1])
        with d1:
            st.date_input("DATE", key=k(f"s{i}_t2_date"), on_change=make_sync_callback(k(f"s{i}_t2_date"), k(f"s{i}_t1_date")))
        with d2:
            st.selectbox("TYPE", ["BUY", "SELL"], key=k(f"s{i}_t2_type"), on_change=make_opposite_callback(k(f"s{i}_t2_type"), k(f"s{i}_t1_type")))
        with d3:
            st.text_input("SYMBOL", key=k(f"s{i}_t2_sym"))
        with d4:
            st.number_input("OPEN", value=0.0, format="%.5f", key=k(f"s{i}_t2_open"))
        with d5:
            st.metric("ACTUAL", data["t2_err"] if data["t2_err"] else f"{data['t2_act']:.5f}")
        with d6:
            st.number_input("SWAP/D", value=0.0, format="%.2f", key=k(f"s{i}_t2_swap"))
        with d7:
            st.number_input("PIPV", value=0.0, format="%.2f", key=k(f"s{i}_t2_pipv"), on_change=make_sync_callback(k(f"s{i}_t2_pipv"), k(f"s{i}_t1_pipv")))
        with d8:
            display_colored_value(data["t2_val"], data["t2_err"])

    # ---- charts
    st.markdown("### Grafico Riferimento (DISPY / CTRV)")
    fig_ref = go.Figure()
    fig_ref.add_trace(go.Scatter(x=[min_x, max_x], y=[current_dispy, current_dispy], mode="lines", name="DISPY", line=dict(color="black", dash="dash", width=2)))
    fig_ref.add_trace(go.Scatter(x=[min_x, max_x], y=[current_dispy, current_ctrv], mode="lines", name="CTRV", line=dict(color="black", width=3)))
    fig_ref.update_yaxes(range=[13600, 21000], title_text="Valore Riferimento")
    fig_ref.update_xaxes(title_text="Scala Temporale", tickformat="%Y-%m-%d")
    fig_ref.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_ref, use_container_width=True)

    st.markdown("---")

    st.markdown("### Grafico Andamento Slot")
    g2_filter = st.selectbox(
        "Filtra Slot:",
        (
            "Visualizza solo slot utilizzati",
            "Visualizza solo slot da 1 a 5",
            "Visualizza solo slot da 6 a 10",
            "Visualizza solo slot da 11 a 15",
        ),
        key=k("g2_filter"),
    )

    if current_chart_data:
        fig_slots = go.Figure()
        for trace in current_chart_data:
            slot_num = int(trace["name"].split(" ")[1])
            show_trace = (
                (g2_filter == "Visualizza solo slot utilizzati")
                or ("1 a 5" in g2_filter and 1 <= slot_num <= 5)
                or ("6 a 10" in g2_filter and 6 <= slot_num <= 10)
                or ("11 a 15" in g2_filter and 11 <= slot_num <= 15)
            )
            if show_trace:
                fig_slots.add_trace(go.Scatter(x=trace["df"]["Date"], y=trace["df"]["Value"], mode="lines", name=trace["name"], line=dict(color=trace["color"], width=3)))

        add_zero_line(fig_slots, min_x, max_x)
        fig_slots.update_yaxes(range=[-3400, 3400], title_text="Valore Monetario Slot")
        fig_slots.update_xaxes(title_text="Scala Temporale", tickformat="%Y-%m-%d")
        fig_slots.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_slots, use_container_width=True)
    else:
        st.info("Compila almeno uno slot per vedere il grafico delle performance.")

    st.markdown("---")

    st.markdown("### Grafico Totale Esposizione")
    if current_chart_data:
        master_dates = np.sort(pd.concat([t["df"]["Date"] for t in current_chart_data]).unique())
        total_pnl_all = np.zeros(len(master_dates))

        for trace in current_chart_data:
            df_temp = trace["df"].set_index("Date")
            aligned = df_temp.reindex(master_dates).ffill().fillna(0)
            total_pnl_all += aligned["Value"].values

        total_df_all = pd.DataFrame({"Date": master_dates, "Total": total_pnl_all})
        fig_total = go.Figure()

        fig_total.add_trace(go.Scatter(x=total_df_all["Date"], y=total_df_all["Total"], mode="lines", name="Totale (1-15)", line=dict(color="#FF0000", width=4)))
        if current_group_totals.get("A") is not None:
            fig_total.add_trace(go.Scatter(x=current_group_totals["A"]["Date"], y=current_group_totals["A"]["Value"], mode="lines", name="Totale (1-5)", line=dict(color="#FFD800", width=3)))
        if current_group_totals.get("B") is not None:
            fig_total.add_trace(go.Scatter(x=current_group_totals["B"]["Date"], y=current_group_totals["B"]["Value"], mode="lines", name="Totale (6-10)", line=dict(color="#0000FF", width=3)))
        if current_group_totals.get("C") is not None:
            fig_total.add_trace(go.Scatter(x=current_group_totals["C"]["Date"], y=current_group_totals["C"]["Value"], mode="lines", name="Totale (11-15)", line=dict(color="#00FF00", width=3)))

        add_zero_line(fig_total, min_x, max_x)
        fig_total.update_yaxes(range=[-3400, 3400], title_text="Valore Totale")
        fig_total.update_xaxes(title_text="Scala Temporale", tickformat="%Y-%m-%d")
        fig_total.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("---")

    st.markdown("### Grafico 4: Analisi Gruppi (1-5, 6-10, 11-15)")
    g4_view = st.selectbox(
        "Seleziona Visualizzazione:",
        (
            "Tutti e 3 i gruppi (Base)",
            "Visualizza solo slot da 1 a 5",
            "Visualizza solo slot da 6 a 10",
            "Visualizza solo slot da 11 a 15",
            "Visualizza slot da 1 a 5 e da 6 a 10 (Combinato)",
            "Visualizza slot da 1 a 5 e da 11 a 15 (Combinato)",
            "Visualizza slot da 6 a 10 e da 11 a 15 (Combinato)",
        ),
        key=k("g4_filter"),
    )

    fig_groups = go.Figure()

    def plot_group_trace(df, name, color, width=4):
        if df is None:
            return
        fig_groups.add_trace(go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name=name, line=dict(color=color, width=width)))

    if "Tutti e 3" in g4_view:
        plot_group_trace(current_group_totals.get("A"), "Gruppo 1-5", "#FFD800", 4)
        plot_group_trace(current_group_totals.get("B"), "Gruppo 6-10", "#1C39BB", 4)
        plot_group_trace(current_group_totals.get("C"), "Gruppo 11-15", "#00FF00", 4)
    elif "1 a 5" in g4_view and "Combinato" not in g4_view:
        plot_group_trace(current_group_totals.get("A"), "Gruppo 1-5", "#FFD800", 4)
    elif "6 a 10" in g4_view and "Combinato" not in g4_view:
        plot_group_trace(current_group_totals.get("B"), "Gruppo 6-10", "#1C39BB", 4)
    elif "11 a 15" in g4_view and "Combinato" not in g4_view:
        plot_group_trace(current_group_totals.get("C"), "Gruppo 11-15", "#00FF00", 4)
    elif "1 a 5 e da 6 a 10" in g4_view:
        plot_group_trace(current_group_totals.get("A"), "Gruppo 1-5", "#FFD800", 3)
        plot_group_trace(current_group_totals.get("B"), "Gruppo 6-10", "#1C39BB", 3)
        if current_group_totals.get("A") is not None and current_group_totals.get("B") is not None:
            df_a = current_group_totals["A"].set_index("Date")
            df_b = current_group_totals["B"].set_index("Date")
            idx_union = df_a.index.union(df_b.index)
            sum_val = df_a.reindex(idx_union).ffill().fillna(0).add(df_b.reindex(idx_union).ffill().fillna(0), fill_value=0)
            plot_group_trace(pd.DataFrame({"Date": sum_val.index, "Value": sum_val.values}), "Combinato (1-10)", "#FFFF00", 5)
    elif "1 a 5 e da 11 a 15" in g4_view:
        plot_group_trace(current_group_totals.get("A"), "Gruppo 1-5", "#FFD800", 3)
        plot_group_trace(current_group_totals.get("C"), "Gruppo 11-15", "#00FF00", 3)
        if current_group_totals.get("A") is not None and current_group_totals.get("C") is not None:
            df_a = current_group_totals["A"].set_index("Date")
            df_c = current_group_totals["C"].set_index("Date")
            idx_union = df_a.index.union(df_c.index)
            sum_val = df_a.reindex(idx_union).ffill().fillna(0).add(df_c.reindex(idx_union).ffill().fillna(0), fill_value=0)
            plot_group_trace(pd.DataFrame({"Date": sum_val.index, "Value": sum_val.values}), "Combinato (1-5 + 11-15)", "#7B1B02", 5)
    elif "6 a 10 e da 11 a 15" in g4_view:
        plot_group_trace(current_group_totals.get("B"), "Gruppo 6-10", "#1C39BB", 3)
        plot_group_trace(current_group_totals.get("C"), "Gruppo 11-15", "#00FF00", 3)
        if current_group_totals.get("B") is not None and current_group_totals.get("C") is not None:
            df_b = current_group_totals["B"].set_index("Date")
            df_c = current_group_totals["C"].set_index("Date")
            idx_union = df_b.index.union(df_c.index)
            sum_val = df_b.reindex(idx_union).ffill().fillna(0).add(df_c.reindex(idx_union).ffill().fillna(0), fill_value=0)
            plot_group_trace(pd.DataFrame({"Date": sum_val.index, "Value": sum_val.values}), "Combinato (6-15)", "#6F00FF", 5)

    add_zero_line(fig_groups, min_x, max_x)
    fig_groups.update_yaxes(range=[-3400, 3400], title_text="Valore Totale Gruppo")
    fig_groups.update_xaxes(title_text="Scala Temporale", tickformat="%Y-%m-%d")
    fig_groups.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig_groups, use_container_width=True)
