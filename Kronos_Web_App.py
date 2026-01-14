import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from scipy.signal import stft, find_peaks, detrend
from scipy.fft import fft, fftfreq
import io
import os
import glob

# --- DIRECTORY CONFIG ---
# 1. Local Absolute Path (Primary for your PC)
LOCAL_DIR = r'C:\Users\hanii\Documents\0. JUPITER SATURNUS\0. TA JEMBATAN\3. data_csv'
# 2. Relative Path (Fallback for Cloud/GitHub)
REPO_DIR = 'data'

if os.path.exists(LOCAL_DIR):
    AUTO_LOAD_DIR = LOCAL_DIR
elif os.path.exists(REPO_DIR):
    AUTO_LOAD_DIR = REPO_DIR
else:
    AUTO_LOAD_DIR = None

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Kronos Analysis",
    page_icon="⚡",
    layout="wide", # Wide layout for better "one window" interactivity
    initial_sidebar_state="expanded"
)

# --- SCIENTIFIC JOURNAL CSS (MINIMALIST) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #ffffff;
        color: #212529;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #000000;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
    }
    p, label, div {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #6c757d !important;
        font-family: 'Arial', sans-serif;
        font-size: 0.9em;
    }
    div[data-testid="stMetricValue"] {
        color: #2c3e50 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px;
        font-family: 'Arial', sans-serif;
    }
    .stButton > button:hover {
        background-color: #1a252f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## Configuration Parameters")
    st.info("Ensure parameters match the sensor specifications.")
    sampling_rate = st.number_input("Sampling Rate (Hz)", value=128.0, step=1.0)
    mass_assumption = st.number_input("Structural Mass (kg)", value=1000.0, step=100.0)
    peak_threshold = st.slider("Peak Detection Threshold", 0.01, 1.0, 0.05)
    st.markdown("---")
    st.caption("Kronos Analysis System v3.0 (Scientific)")

# --- CORE LOGIC ---
def analyze_signal_physics(signal, fs, mass, threshold):
    n = len(signal)
    acceleration = detrend(signal) 
    force = acceleration * mass
    fft_vals = fft(acceleration)
    fft_amp = np.abs(fft_vals[:n//2]) * 2 / n
    freqs = fftfreq(n, 1/fs)[:n//2]
    peaks, _ = find_peaks(fft_amp, height=np.max(fft_amp) * threshold, distance=5)
    if len(peaks) > 0:
        idx_max = np.argmax(fft_amp[peaks])
        dom_freq = freqs[peaks[idx_max]]
        dom_amp = fft_amp[peaks[idx_max]]
    else:
        dom_freq = 0
        dom_amp = 0
    # Increased nperseg for better frequency detail (256 -> 512)
    f_stft, t_stft, zxx = stft(acceleration, fs, nperseg=512)
    return {
        'time': np.arange(n) / fs, 'acc': acceleration, 'force': force,
        'freqs': freqs, 'fft_amp': fft_amp, 'stft': (f_stft, t_stft, zxx),
        'metrics': {'max_acc': np.max(np.abs(acceleration)), 'max_force': np.max(np.abs(force)),
                    'dom_freq': dom_freq, 'dom_amp': dom_amp, 'duration': n / fs}
    }

def ingest_csv_data(file_source):
    # Modified to accept both UploadedFile object AND file path string
    try:
        if isinstance(file_source, str):
            # It's a file path
            with open(file_source, 'r') as f:
                lines = f.readlines()
        else:
            # It's a streamlit UploadedFile
            file_source.seek(0)
            content = file_source.getvalue().decode("utf-8")
            lines = content.splitlines()
            file_source.seek(0)

        # Detect Header
        header_row = 0
        for i, line in enumerate(lines):
            if 'DATA_START' in line:
                header_row = i + 1
                break
        
        # Read properly
        if isinstance(file_source, str):
            df = pd.read_csv(file_source, skiprows=header_row)
        else:
            file_source.seek(0)
            df = pd.read_csv(file_source, skiprows=header_row)
        
        if df.shape[1] > 1:
            signal_col = df.columns[1] 
            df.rename(columns={signal_col: 'Signal'}, inplace=True)
            df['Signal'] = pd.to_numeric(df['Signal'], errors='coerce')
            df.dropna(subset=['Signal'], inplace=True)
            return df
        return None
    except Exception as e:
        # st.error(f"Error reading {file_source}: {e}")
        return None

# --- UI ---
st.title("KRONOS ANALYSIS")
st.markdown("#### *Interactive Structural Health Monitoring*")
st.markdown("---")

# Data Loading Section
data_sources = []
source_names = []

# 1. Auto-Load from Folder
if AUTO_LOAD_DIR and os.path.exists(AUTO_LOAD_DIR):
    local_files = glob.glob(os.path.join(AUTO_LOAD_DIR, "*.csv"))
    if local_files:
        st.success(f"📂 Auto-loaded {len(local_files)} files from: `{AUTO_LOAD_DIR}`")
        for f in local_files:
            data_sources.append(f)
            source_names.append(os.path.basename(f))
    else:
        st.warning(f"📂 Folder found but empty: `{AUTO_LOAD_DIR}`")
else:
    if AUTO_LOAD_DIR:
        st.info(f"📂 Auto-load folder not found: `{AUTO_LOAD_DIR}`")
    else:
        st.info("ℹ️ Cloud Mode: Upload files manually via the uploader below.")

# 2. Manual Upload
uploaded_files = st.file_uploader("Attach Additional Data Source (CSV)", type=['csv'], accept_multiple_files=True)
if uploaded_files:
    for f in uploaded_files:
        data_sources.append(f)
        source_names.append(f.name)

# --- PROCESSING ---
if data_sources:
    # Selector to choose which file to view if multiple
    selected_file_index = 0
    if len(data_sources) > 1:
        selected_file_name = st.selectbox("Select File to Analyze:", source_names)
        selected_file_index = source_names.index(selected_file_name)
    
    current_file = data_sources[selected_file_index]
    current_name = source_names[selected_file_index]

    st.markdown(f"### Analysis of Sample: `{current_name}`")
    df = ingest_csv_data(current_file)
    
    if df is not None:
        res = analyze_signal_physics(df['Signal'].values, sampling_rate, mass_assumption, peak_threshold)
        m = res['metrics']
        
        # Metrics
        with st.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Acceleration", f"{m['max_acc']:.3f} m/s²")
            c2.metric("Max Dynamic Force", f"{m['max_force']:.1f} N")
            c3.metric("Dominant Frequency", f"{m['dom_freq']:.3f} Hz")
            c4.metric("Duration", f"{m['duration']:.2f} s")
        
        st.write(" ")
        
        # CONFIG: Enable JPG Download for Plotly
        config = {
            'toImageButtonOptions': {
                'format': 'jpeg',
                'filename': f'Figure_{current_name}',
                'height': 800, 'width': 1200, 'scale': 1.5
            },
            'displaylogo': False
        }
        
        # 1. Time Series
        fig_time = make_subplots(specs=[[{"secondary_y": True}]])
        fig_time.add_trace(go.Scatter(x=res['time'], y=res['acc'], name="Akselerasi", line=dict(color='black', width=0.8)), secondary_y=False)
        fig_time.add_trace(go.Scatter(x=res['time'], y=res['force'], name="Gaya Dalam", line=dict(color='#c0392b', width=0.8, dash='dot')), secondary_y=True)
        fig_time.update_layout(title="Grafik Respon Waktu (Time Domain)", template="simple_white", height=400,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_time.update_yaxes(title_text="Akselerasi (m/s²)", secondary_y=False)
        fig_time.update_yaxes(title_text="Gaya Dalam / Force (N)", secondary_y=True)
        st.plotly_chart(fig_time, use_container_width=True, config=config)
        
        col_l, col_r = st.columns(2)
        
        # 2. FFT
        with col_l:
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=res['freqs'], y=res['fft_amp'], name="Amplitudo", 
                                            line=dict(color='#2c3e50'), fill='tozeroy', fillcolor='rgba(44, 62, 80, 0.1)'))
            fig_fft.update_layout(title="Grafik Frekuensi & Amplitudo (FFT)", template="simple_white", height=350)
            fig_fft.update_xaxes(title_text="Frekuensi (Hz)")
            fig_fft.update_yaxes(title_text="Amplitudo")
            st.plotly_chart(fig_fft, use_container_width=True, config=config)
            
        # 3. Spectrogram
        with col_r:
            # Use 'Blues' colorscale for better clarity as requested
            fig_spec = go.Figure(data=go.Heatmap(z=np.abs(res['stft'][2]), x=res['stft'][1], y=res['stft'][0], colorscale='Blues'))
            fig_spec.update_layout(title="Grafik STFT (Time-Frequency)", template="simple_white", height=350)
            fig_spec.update_xaxes(title_text="Waktu (s)")
            fig_spec.update_yaxes(title_text="Frekuensi (Hz)")
            st.plotly_chart(fig_spec, use_container_width=True, config=config)
        
        # --- JPG REPORT GENERATION (SCIENTIFIC STYLE) ---
        def generate_static_report_image(res, filename):
            # Setup Light Layout
            plt.style.use('default')
            fig_static = plt.figure(figsize=(8.27, 11.69)) # A4
            fig_static.patch.set_facecolor('white')
            
            # Header
            fig_static.text(0.5, 0.95, "KRONOS ANALYSIS REPORT", ha='center', va='center', 
                            fontsize=20, color='#000000', fontfamily='sans-serif', weight='bold')
            fig_static.text(0.5, 0.92, f"Sample: {filename}", ha='center', va='center', 
                            fontsize=10, color='#666', fontfamily='sans-serif')

            # Metrics Table
            md = res['metrics']
            metrics_data = [
                ["Max Acceleration", f"{md['max_acc']:.4f} m/s²"],
                ["Max Dynamic Force", f"{md['max_force']:.2f} N"],
                ["Dominant Freq", f"{md['dom_freq']:.2f} Hz"],
                ["Signal Duration", f"{md['duration']:.2f} s"]
            ]
            
            # Add a table at the top
            ax_table = fig_static.add_axes([0.15, 0.82, 0.7, 0.08])
            ax_table.axis('off')
            table = ax_table.table(cellText=metrics_data, colWidths=[0.4, 0.4], loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.2)

            # Gridspec for plots
            gs = fig_static.add_gridspec(3, 1, top=0.78, bottom=0.05, hspace=0.4, left=0.12, right=0.9)

            # 1. Time Series
            ax1 = fig_static.add_subplot(gs[0, 0])
            ax1.plot(res['time'], res['acc'], 'k-', lw=0.6, label='Acc')
            ax12 = ax1.twinx()
            ax12.plot(res['time'], res['force'], 'r--', lw=0.6, label='Force')
            ax1.set_title("Figure 1: Time Domain Response", fontsize=10, fontfamily='serif', loc='left')
            ax1.set_ylabel("Acceleration (m/s²)", fontsize=8)
            ax12.set_ylabel("Force (N)", fontsize=8, color='red')
            ax1.grid(True, alpha=0.3)

            # 2. FFT
            ax2 = fig_static.add_subplot(gs[1, 0])
            ax2.plot(res['freqs'], res['fft_amp'], color='#2c3e50', lw=1)
            ax2.fill_between(res['freqs'], res['fft_amp'], color='#2c3e50', alpha=0.1)
            ax2.set_title("Figure 2: Frequency Spectrum", fontsize=10, fontfamily='serif', loc='left')
            ax2.set_ylabel("Amplitude", fontsize=8)
            ax2.set_xlim(0, 20)
            ax2.grid(True, alpha=0.3)

            # 3. Spectrogram
            ax3 = fig_static.add_subplot(gs[2, 0])
            f_stft, t_stft, zxx = res['stft']
            c = ax3.pcolormesh(t_stft, f_stft, np.abs(zxx), shading='gouraud', cmap='Blues')
            ax3.set_title("Figure 3: Spectrogram", fontsize=10, fontfamily='serif', loc='left')
            ax3.set_ylabel("Frequency (Hz)", fontsize=8)
            ax3.set_xlabel("Time (s)", fontsize=8)
            ax3.set_ylim(0, 20)
            # Add colorbar for detail
            plt.colorbar(c, ax=ax3, label='Magnitude')
            
            # Buffer
            buf = io.BytesIO()
            fig_static.savefig(buf, format='jpeg', dpi=150, bbox_inches='tight')
            plt.close(fig_static)
            return buf.getvalue()

        st.write(" ")
        if st.button(f"📸 Generate Graphic Report ({current_name})"):
            with st.spinner("Generating Report..."):
                img_data = generate_static_report_image(res, current_name)
                st.image(img_data, caption="Static Report Preview", use_container_width=True)
                st.download_button(
                    label="💾 Download JPG Report",
                    data=img_data,
                    file_name=f"Report_{current_name}.jpg",
                    mime="image/jpeg"
                )

        st.markdown("---")
        
else:
    st.info("Waiting for data source...")
