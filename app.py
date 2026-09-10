import streamlit as st
import time
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import plotly.express as px

from src.vision_engine import get_waste_composition_from_results
from src.weather_api import get_weather_context
from src.energy_math import calculate_energy_potential

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI-004 WtE Manager",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # preserving your specific model path
    return YOLO('weights/wasteland_model/WtE_Predictor/v3_final_refinement/weights/best.pt')

model = load_model()

# --- SIDEBAR: HISTORY ---
with st.sidebar:
    st.title("🗂️ Batch History")
    st.markdown("---")
    
    if len(st.session_state['history']) > 0:
        history_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(
            history_df[['Time', 'Decision', 'LHV']], 
            use_container_width=True,
            hide_index=True
        )
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv_data, "report.csv", "text/csv")
    else:
        st.info("No scans performed today.")
    
    st.markdown("---")
    st.caption("v2.1 | Model: YOLOv11n | Region: Vellore")

# --- HEADER: WEATHER CONTEXT ---
weather = get_weather_context() 

st.markdown("### 🌤️ Environmental Context (Vellore)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", "32°C", "1.2°C")
col2.metric("Humidity", f"{weather['humidity']}%", "5%" if weather['is_monsoon'] else "-2%")
col3.metric("Rainfall (1h)", f"{weather['rain_1h']} mm")
status_color = "off" if not weather['is_monsoon'] else "normal"
col4.metric("Condition", "MONSOON" if weather['is_monsoon'] else "DRY SEASON", delta_color=status_color)
st.divider()

# --- MAIN INTERFACE ---
st.subheader("📷 Waste Analysis Terminal")

uploaded_file = st.file_uploader("Upload Batch Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Save file
    with open("temp_scan.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- VISUALIZATION SECTION ---
    col_raw, col_ai = st.columns(2)
    
    with col_raw:
        st.caption("Step 1: Raw Input Feed")
        raw_image = Image.open("temp_scan.jpg")
        st.image(raw_image, use_container_width=True, channels="RGB")

    with st.spinner("⚡ AI Processing: Segmenting Objects & Calculating Thermodynamics..."):
        # 1. Vision Analysis
        # Preserving your low confidence threshold and plot settings
        results = model.predict("temp_scan.jpg", conf=0.0001)
        annotated_frame = results[0].plot(line_width=1, font_size=0.2, labels=True, masks=False)
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(annotated_rgb)
        
        # 2. Composition Calculation
        comp = get_waste_composition_from_results(results[0], model.names)
        
        # 3. Advanced Energy Math (Returns Tuple now)
        total_lhv, breakdown_data = calculate_energy_potential(comp, weather)

    with col_ai:
        st.caption(f"Step 2: AI Segmentation Result (Found {len(breakdown_data)} Categories)")
        st.image(final_image, use_container_width=True)
        st.toast("Analysis Complete!", icon="✅")

    st.markdown("---")

    # --- ADVANCED ANALYTICS ---
    st.subheader("📊 Operational Analytics")
    
    chart_col, data_col = st.columns([1, 1.5])
    
    # A. Constitutional % (Donut Chart)
    with chart_col:
        st.markdown("**Batch Composition (Mass %)**")
        df_comp = pd.DataFrame(list(comp.items()), columns=['Material', 'Percentage'])
        
        # Using Plotly Express for the interactive Donut Chart
        fig = px.pie(
            df_comp, 
            values='Percentage', 
            names='Material', 
            hole=0.4, 
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # B. Energy Yield Breakdown (The Logic Table)
    with data_col:
        st.markdown("**Energy Potential Breakdown**")
        
        df_breakdown = pd.DataFrame(breakdown_data)
        
        if not df_breakdown.empty:
            st.dataframe(
                df_breakdown[['Material', 'Composition', 'Moisture', 'Dry Potential', 'Actual Energy']],
                column_config={
                    "Material": "Waste Type",
                    "Composition": st.column_config.NumberColumn("Mass %", format="%.1f%%"),
                    # Visual Progress Bar for Moisture
                    "Moisture": st.column_config.ProgressColumn(
                        "Moisture", 
                        format="%.0f%%", 
                        min_value=0, 
                        max_value=100
                    ),
                    "Dry Potential": st.column_config.NumberColumn("Theoretical LHV", format="%.1f MJ"),
                    "Actual Energy": st.column_config.NumberColumn("Real LHV (Wet)", format="%.1f MJ"),
                },
                hide_index=True,
                use_container_width=True
            )
            st.caption("*Theoretical LHV = Energy if dry. | Real LHV = Energy after rain penalty.*")
        else:
            st.warning("No waste detected or composition is empty.")

    # --- FINAL DECISION BANNER ---
    st.divider()
    
    # Big Metric at the bottom
    col_metric, col_decision = st.columns([1, 2])
    
    with col_metric:
        st.metric("🔥 NET CALORIFIC VALUE", f"{total_lhv:.2f} MJ/kg", delta="Final Yield")
    
    with col_decision:
        if total_lhv > 7.5:
            st.success("### ✅ ACTION: DIRECT COMBUSTION\n**Reasoning:** High caloric plastic/paper content offsets moisture. Safe for boiler.")
            decision = "Incinerate"
        elif total_lhv > 4.0:
            st.warning("### ⚠️ ACTION: PRE-DRYING REQUIRED\n**Reasoning:** Energy viable, but moisture > 40%. Dry for 24h to prevent boiler corrosion.")
            decision = "Pre-Dry"
        else:
            st.error("### 🚨 ACTION: REJECT / BIOGAS\n**Reasoning:** Energy negative or neutral. Incineration will consume diesel. Divert to composting.")
            decision = "Biogas"

    # History Logic
    if 'last_file' not in st.session_state or st.session_state['last_file'] != uploaded_file.name:
        timestamp = time.strftime("%H:%M:%S")
        st.session_state['history'].insert(0, {
            "Time": timestamp,
            "Decision": decision,
            "LHV": f"{total_lhv:.2f}"
        })
        st.session_state['last_file'] = uploaded_file.name
