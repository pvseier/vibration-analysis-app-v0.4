
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
import io

st.set_page_config(page_title="Vibration Analysis Full App", layout="wide")
st.title("Vibration Velocity + FFT Analysis App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, header=None)
    df_raw.columns = ['Time_s', 'Accel_X_g', 'Accel_Y_g', 'Accel_Z_g']
    st.success("File loaded successfully.")

    st.sidebar.header("Filter Settings")
    lowcut = st.sidebar.slider("Low Cut Frequency (Hz)", 0.1, 20.0, 2.0)
    highcut = st.sidebar.slider("High Cut Frequency (Hz)", 20.0, 200.0, 99.0)
    order = st.sidebar.selectbox("Filter Order", [2, 4, 6], index=1)

    st.sidebar.header("Time Range (for viewing only)")
    view_start = st.sidebar.number_input("Start Time (s)", value=float(df_raw['Time_s'].min()))
    view_end = st.sidebar.number_input("End Time (s)", value=float(df_raw['Time_s'].max()))

    st.sidebar.header("FFT Frequency Range (Hz)")
    fft_start = st.sidebar.number_input("FFT Start Frequency (Hz)", value=0.0, min_value=0.0, step=1.0)
    fft_end = st.sidebar.number_input("FFT End Frequency (Hz)", value=200.0, min_value=1.0, step=1.0)

    st.sidebar.header("Graph Titles")
    velocity_title = st.sidebar.text_input("Velocity Graph Title", value="Filtered Velocity (DC Offset Removed)")
    fft_title = st.sidebar.text_input("FFT Graph Title", value="FFT of DC-Free Velocity")

    df = df_raw.copy()
    g_to_in_s2 = 386.09
    for axis in ['X', 'Y', 'Z']:
        df[f'Accel_{axis}_in_s2'] = df[f'Accel_{axis}_g'] * g_to_in_s2

    fs = 1 / df['Time_s'].diff().mean()

    def butter_bandpass_filter(data, lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    for axis in ['X', 'Y', 'Z']:
        df[f'Filtered_Accel_{axis}'] = butter_bandpass_filter(df[f'Accel_{axis}_in_s2'], lowcut, highcut, fs, order)
        df[f'Velocity_{axis}'] = cumulative_trapezoid(df[f'Filtered_Accel_{axis}'], df['Time_s'], initial=0)
        df[f'DC_Free_Velocity_{axis}'] = df[f'Velocity_{axis}'] - df[f'Velocity_{axis}'].mean()

    df_view = df[(df['Time_s'] >= view_start) & (df['Time_s'] <= view_end)]

    st.subheader("Velocity Plot")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df_view['Time_s'], df_view['DC_Free_Velocity_X'], label='Velocity X (in/s)')
    ax1.plot(df_view['Time_s'], df_view['DC_Free_Velocity_Y'], label='Velocity Y (in/s)')
    ax1.plot(df_view['Time_s'], df_view['DC_Free_Velocity_Z'], label='Velocity Z (in/s)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity (in/s)")
    ax1.set_title(velocity_title)
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("FFT Spectrum of Velocity")
    def compute_fft(signal, fs):
        N = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freqs = np.fft.rfftfreq(N, d=1/fs)
        magnitude = np.abs(fft_vals)
        return fft_freqs, magnitude

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for axis, label in zip(['X', 'Y', 'Z'], ['X', 'Y', 'Z']):
        freqs, mags = compute_fft(df[f'DC_Free_Velocity_{axis}'], fs)
        ax2.plot(freqs, mags, label=label)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(fft_title)
    ax2.set_xlim([fft_start, fft_end])
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

    # Peak velocity based on view range
    peak_velocity = df_view[[f'DC_Free_Velocity_{axis}' for axis in ['X', 'Y', 'Z']]].abs().max().max()
    st.metric("Max Peak Velocity (in view range)", f"{peak_velocity:.3f} in/s")

    with st.expander("Download Data and Graphs"):
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "processed_vibration_data.csv", "text/csv")

        try:
            import xlsxwriter
            excel_output = io.BytesIO()
            with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='ProcessedData')
            st.download_button("Download Excel", excel_output.getvalue(), "processed_vibration_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.warning(f"Excel download failed: {e}")

        for fig, name in zip([fig1, fig2], ["velocity_plot", "fft_plot"]):
            img_buf_png = io.BytesIO()
            fig.savefig(img_buf_png, format="png")
            st.download_button(f"Download {name}.png", img_buf_png.getvalue(), f"{name}.png", "image/png")

            img_buf_jpeg = io.BytesIO()
            fig.savefig(img_buf_jpeg, format="jpeg")
            st.download_button(f"Download {name}.jpeg", img_buf_jpeg.getvalue(), f"{name}.jpeg", "image/jpeg")
