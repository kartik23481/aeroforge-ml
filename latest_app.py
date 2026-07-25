
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import percentileofscore
import requests

from mlops_src.utils.feature_utils import (
    is_same_region, part_of_month, part_of_day,
    make_month_object, direct_flight,
    duration_category
)

RENDER_API_URL = "https://testing-dv-dashub.onrender.com/predict"
RENDER_HEALTH_URL = "https://testing-dv-dashub.onrender.com/healthz"

def call_backend(payload):
    try:
        res = requests.post(RENDER_API_URL, json=payload, timeout=20)
        if res.status_code == 200:
            return res.json().get("predicted_price")
        else:
            st.error(res.text)
            return None
    except Exception as e:
        st.error(str(e))
        return None

st.set_page_config(
    page_title="AeroForge ML | Flight Pricing",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif !important;
        }

        /* Deep Slate Background */
        .stApp {
            background-color: #0b0f19;
        }

        /* Elevate the Form into a SaaS Card */
        [data-testid="stForm"] {
            background-color: #111827 !important;
            border: 1px solid #1f2937 !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.2) !important;
        }

        /* Style the inputs targeting Base Web UI */
        div[data-baseweb="input"] > div, 
        div[data-baseweb="select"] > div, 
        div[data-baseweb="base-input"] {
            background-color: #1f2937 !important;
            border: 1px solid #374151 !important;
            border-radius: 8px !important;
            color: #f3f4f6 !important;
            transition: all 0.2s ease;
        }

        /* Input Focus Glow */
        div[data-baseweb="input"] > div:focus-within, 
        div[data-baseweb="select"] > div:focus-within {
            border-color: #0ea5e9 !important;
            box-shadow: 0 0 0 1px #0ea5e9 !important;
        }

        /* Soften Labels */
        label, .st-emotion-cache-1y4p8pa {
            color: #9ca3af !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            letter-spacing: 0.025em !important;
        }

        /* Force Gradient on Submit Button via data-testid */
        [data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            padding: 0.75rem 2rem !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }

        [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.4) !important;
        }

        /* Result Box Formatting */
        .result-box {
            background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-left: 4px solid #06b6d4;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            margin: 2rem auto;
            max-width: 600px;
        }
        .price-label {
            color: #9ca3af;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 0.5rem;
        }
        .price-value {
            font-size: 48px;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            text-shadow: 0 0 20px rgba(6, 182, 212, 0.3);
        }
        .section-header {
            color: #ffffff;
            font-size: 20px;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 1px solid #1f2937;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; margin-top: 2rem;'>
        <h1 style='color: #ffffff; font-size: 42px; margin-bottom: 0;'>✈️ AeroForge ML</h1>
        <p style='color: #00d2ff; font-size: 16px; letter-spacing: 1px; font-weight: 500;'>ROUTE AWARE PREDICTIVE PRICING SYSTEM</p>
    </div>
""", unsafe_allow_html=True)


if 'backend_awake' not in st.session_state:
    startup_container = st.empty()
    with startup_container.container():
        st.markdown("""
            <div style='background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 3rem; text-align: center; max-width: 600px; margin: 0 auto; box-shadow: 0 10px 30px rgba(0,0,0,0.5);'>
                <h3 style='color: #00d2ff; margin-bottom: 15px;'>🔌 Initializing Inference Engine</h3>
                <p style='color: #8b9eb3; font-size: 15px; margin-bottom: 25px;'>"Establishing a secure connection to the inference engine. Allocating serverless resources typically takes 30–50 seconds. Thank you for your patience."</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Connecting to Backend..."):
            try:
                # 120s timeout ensures Streamlit doesn't crash while Render is booting
                requests.get(RENDER_HEALTH_URL, timeout=120) 
                st.session_state['backend_awake'] = True
            except Exception as e:
                st.error("Failed to connect to the backend. The server might be down or updating.")
                st.stop()
                
    startup_container.empty()



def apply_plotly_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b9eb3"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

@st.cache_data
def load_training_data():
    data_path = "data/processed/train_data.csv"
    try:
        train_df = pd.read_csv(data_path)
        train_df['source'] = train_df['source'].str.lower()
        train_df['destination'] = train_df['destination'].str.lower()
        return train_df
    except:
        return pd.DataFrame()

train_df = load_training_data()

airline_list = [
    "Indigo", "Air India", "Jet Airways", "Spicejet",
    "Multiple Carriers", "Goair", "Vistara", "Air Asia", "Trujet"
]
source_list = ["delhi", "banglore", "mumbai", "chennai", "kolkata"]
destination_list = ["cochin", "banglore", "delhi", "new delhi", "hyderabad","kolkata"]
additional_info_list = [
    "no info", "in-flight meal not included", "no check-in baggage included",
    "1 long layover", "change airports", "business class",
    "1 short layover", "red-eye flight"
]

def create_deal_gauge(price, historical_prices):
    if historical_prices.empty:
        return None
    perc = percentileofscore(historical_prices, price)
    if perc <= 25:
        deal_text, color = "Exceptional Value", "#00e676"
    elif perc <= 50:
        deal_text, color = "Fair Market Price", "#00d2ff"
    elif perc <= 75:
        deal_text, color = "Slightly Elevated", "#ffab00"
    else:
        deal_text, color = "Premium Pricing", "#ff3d00"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        number={'prefix': "₹", 'font': {'size': 36, 'color': '#ffffff'}},
        delta={'reference': historical_prices.mean(), 'relative': False, 'valueformat': '.0f'},
        title={'text': f"<span style='font-size:20px; color:{color}'>{deal_text}</span><br><span style='font-size:12px; color:#8b9eb3'>vs. Route Avg: ₹{historical_prices.mean():,.0f}</span>"},
        gauge={
            'axis': {'range': [historical_prices.min(), historical_prices.max()], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [historical_prices.min(), historical_prices.quantile(0.25)], 'color': "rgba(0, 230, 118, 0.1)"},
                {'range': [historical_prices.quantile(0.75), historical_prices.max()], 'color': "rgba(255, 61, 0, 0.1)"}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=320, margin=dict(t=50, b=20)
    )
    return fig



with st.form("flight_input_form1"):
    st.markdown("<div class='section-header'>Configure Flight Parameters</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: airline = st.selectbox("Airline Provider", airline_list)
    with col2: source = st.selectbox("Departure City", source_list)
    with col3: destination = st.selectbox("Arrival City", destination_list)
    
    col4, col5, col6 = st.columns(3)
    with col4: duration = st.number_input("Total Duration (mins)", min_value=0, value=180)
    with col5: total_stops = st.number_input("Layover Count", min_value=0, value=0)
    with col6: additional_info = st.selectbox("Special Constraints", additional_info_list)
    
    col7, col8 = st.columns(2)
    with col7: date_input = st.date_input("Scheduled Date", value=datetime(2019, 6, 1))
    with col8: dept_time_hour = st.number_input("Departure Hour (24H)", min_value=0, max_value=23, value=10)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Execute Prediction Model", type="primary", use_container_width=True)


if submitted:
    st.session_state['payload'] = {
        "airline": airline, "source": source, "destination": destination,
        "duration": int(duration), "total_stops": int(total_stops),
        "additional_info": additional_info, "dep_time_hour": int(dept_time_hour),
        "date": str(date_input)
    }
    
    
    prediction = call_backend(st.session_state['payload'])
    if prediction is not None:
        st.session_state['prediction'] = prediction
        st.session_state['show_results'] = True
        st.session_state['animate'] = True # 
    else:
        st.stop()


if st.session_state.get('show_results', False):
    payload = st.session_state['payload']
    prediction = st.session_state['prediction']

    
    result_placeholder = st.empty()
    
    
    if st.session_state.get('animate', False):
        step_size = max(1, int(prediction)//30) 
        for i in range(0, int(prediction)+1, step_size):
            result_placeholder.markdown(f"""
                <div class='result-box'>
                    <p class='price-label'>Algorithmic Price Estimation</p>
                    <p class='price-value'>₹ {i:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.01)
        st.session_state['animate'] = False
        

    result_placeholder.markdown(f"""
        <div class='result-box'>
            <p class='price-label'>Algorithmic Price Estimation</p>
            <p class='price-value' style='color: #00d2ff;'>₹ {prediction:,.2f}</p>
        </div>
    """, unsafe_allow_html=True)


    if not train_df.empty:
        route_prices = train_df[
            (train_df['source'] == payload['source'].lower()) & (train_df['destination'] == payload['destination'].lower())
        ]['price']
        if not route_prices.empty:
            gauge_fig = create_deal_gauge(prediction, route_prices)
            if gauge_fig:
                st.plotly_chart(gauge_fig, use_container_width=True)


    st.markdown("<div class='section-header'>Market Competitor Analysis</div>", unsafe_allow_html=True)
    

    if 'airline_df' not in st.session_state or st.session_state.get('animate', True):
        airline_predictions = []
        for air in airline_list:
            temp_payload = payload.copy()
            temp_payload["airline"] = air
            pred = call_backend(temp_payload)
            if pred: airline_predictions.append((air, pred))
        st.session_state['airline_df'] = pd.DataFrame(airline_predictions, columns=['Airline', 'Predicted Price'])
            
    airline_df = st.session_state['airline_df']
    fig = px.bar(airline_df, x='Airline', y='Predicted Price', color_discrete_sequence=['#0072ff'])
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    best_airline = airline_df.loc[airline_df['Predicted Price'].idxmin()]
    st.markdown(f"""
        <div style="background: rgba(0, 230, 118, 0.1); border-left: 4px solid #00e676; padding: 1rem; border-radius: 4px; color: #ffffff;">
            Optimal Carrier: <b>{best_airline['Airline']}</b> offers the lowest estimated fare at <b>₹{best_airline['Predicted Price']:,.2f}</b>.
        </div>
    """, unsafe_allow_html=True)


    st.markdown("<div class='section-header'>Feature Influence Analytics</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕒 Temporal Variance", "📅 Date Elasticity", "🚥 Layover Impact", "🗺️ Spatial Routing", "📊 Historical Context"
    ])

    with tab1:
        time_buckets = [(6, "Morning"), (12, "Afternoon"), (18, "Evening"), (22, "Night")]
        time_predictions = [{"Part of Day": label, "Predicted Price": call_backend({**payload, "dep_time_hour": hour})} for hour, label in time_buckets]
        fig2 = apply_plotly_theme(px.bar(pd.DataFrame(time_predictions), x='Part of Day', y='Predicted Price', color_discrete_sequence=['#00d2ff']))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='info-panel'>Price volatility based on departure window segments.</div>", unsafe_allow_html=True)

    with tab2:
        date_predictions = []
        current_date = datetime.strptime(payload["date"], "%Y-%m-%d").date()
        for i in range(-3, 4):
            new_date = current_date + pd.Timedelta(days=i)
            pred = call_backend({**payload, "date": str(new_date)})
            if pred: date_predictions.append({"Date": new_date.strftime("%Y-%m-%d"), "Predicted Price": pred})
        fig3 = apply_plotly_theme(px.line(pd.DataFrame(date_predictions), x='Date', y='Predicted Price', markers=True, color_discrete_sequence=['#00d2ff']))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<div class='info-panel'>Fare elasticity spanning ±3 days from your selected date.</div>", unsafe_allow_html=True)

    with tab3:
        stops_predictions = [{"Stops": f"{s} Layover(s)", "Predicted Price": call_backend({**payload, "total_stops": s})} for s in [0, 1, 2]]
        fig4 = apply_plotly_theme(px.bar(pd.DataFrame(stops_predictions), x='Stops', y='Predicted Price', color_discrete_sequence=['#00d2ff']))
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        source_predictions = [{"Source": src.title(), "Predicted Price": call_backend({**payload, "source": src})} for src in source_list]
        fig5 = apply_plotly_theme(px.bar(pd.DataFrame(source_predictions), x='Source', y='Predicted Price', color_discrete_sequence=['#00d2ff']))
        st.plotly_chart(fig5, use_container_width=True)

    with tab5:
        if not train_df.empty:
            st.markdown("<p style='color:#ffffff; font-weight:600; font-size:18px;'>1. Origin-Destination Mapping</p>", unsafe_allow_html=True)
            
            payload_date = datetime.strptime(payload["date"], "%Y-%m-%d")
            user_month, month_name = payload_date.month, payload_date.strftime("%B")

            filter_by_month = st.toggle(f"Isolate data to {month_name}", value=False)
            
            route_data = train_df[(train_df['source'] == payload['source'].lower()) & (train_df['destination'] == payload['destination'].lower())]
            
            if filter_by_month: 
                route_data = route_data[route_data['dtoj_month'] == user_month]
            
            if not route_data.empty:
                route_data['total_stops'] = route_data['total_stops'].astype(str)
                fig6 = px.scatter(route_data, x='duration', y='price', color='total_stops', opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set2)
                fig6 = apply_plotly_theme(fig6)
                fig6.add_trace(go.Scatter(x=[payload['duration']], y=[prediction], name='Your Prediction', mode='markers', marker=dict(color='#FFD700', size=16, line=dict(color='white', width=2), symbol='star')))
                st.plotly_chart(fig6, use_container_width=True)

            st.markdown("<br><p style='color:#ffffff; font-weight:600; font-size:18px;'>2. Global Duration Clustering</p>", unsafe_allow_html=True)
            duration_margin = 50
            payload_dur = payload['duration']
            similar_flights_data = train_df[(train_df['duration'] >= payload_dur - duration_margin) & (train_df['duration'] <= payload_dur + duration_margin)]
            
            if not similar_flights_data.empty:
                similar_flights_data['total_stops'] = similar_flights_data['total_stops'].astype(str)
                fig7 = px.scatter(similar_flights_data, x='duration', y='price', color='total_stops', opacity=0.4, color_discrete_sequence=px.colors.qualitative.Set2)
                fig7 = apply_plotly_theme(fig7)
                fig7.add_trace(go.Scatter(x=[payload_dur], y=[prediction], name='Your Prediction', mode='markers', marker=dict(color='#FFD700', size=16, line=dict(color='white', width=2), symbol='star')))
                st.plotly_chart(fig7, use_container_width=True)
                st.markdown("<div class='info-panel'>The Gold Star represents your current prediction against historical clusters.</div>", unsafe_allow_html=True)