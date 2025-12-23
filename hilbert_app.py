import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import hilbert

# --- Page Config ---
st.set_page_config(
    page_title="Hilbert Transform Explorer",
    page_icon="📡",
    layout="wide"
)

# --- CSS ---
st.markdown("""
<style>
    .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    .metric-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# st.title("📡 The Hilbert Transform")
st.markdown("### 📡 The Hilbert Transform")
# with st.expander("📋 Instructions"):
#     st.markdown(r"""The **Hilbert Transform** is a 90° phase shifter used to create **Analytic Signals** (one-sided spectrum) and enable **Single Sideband (SSB)** modulation.""")

# --- Helper Functions ---
def design_hilbert_fir(N):
    """
    Design a Hilbert Transformer using Remez algorithm.
    Band edges: 0.05 to 0.45 (normalized to 1.0)
    """
    # Remez expects band edges normalized to Nyquist (0.5)
    # The text suggests a passband avoiding 0 and 0.5.
    # We will use 0.1 to 0.9 of Nyquist.
    h = signal.remez(N, [0.05, 0.45], [1], type='hilbert')
    return h

def plot_spectrum(sig, fs, ax, title, color='b', sides='one'):
    N = len(sig)
    X = np.fft.fft(sig)
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # Shift for plotting
    X_shifted = np.fft.fftshift(X)
    freqs_shifted = np.fft.fftshift(freqs)
    
    mag = 20 * np.log10(np.abs(X_shifted) + 1e-12)
    
    if sides == 'one':
        # Plot only positive frequencies
        mask = freqs_shifted >= 0
        ax.plot(freqs_shifted[mask], mag[mask], color=color)
        ax.set_xlim(0, fs/2)
    else:
        ax.plot(freqs_shifted, mag, color=color)
        ax.set_xlim(-fs/2, fs/2)
        
    ax.set_title(title)
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "1️⃣ The Hilbert Filter",
    "2️⃣ Analytic Signals (Envelope)",
    "3️⃣ Single Sideband (SSB)"
])

# ==============================================================================
# TAB 1: THE FILTER
# ==============================================================================
with tab1:
    # st.header("1. The Hilbert Filter")
    with st.expander("📋 Instructions"):
        st.markdown(r"""
        The ideal Hilbert Transformer has an impulse response $h[n] = \frac{2}{\pi n}$ for odd $n$ (0 otherwise).
        It shifts positive frequencies by $-90^\circ$ (multiply by $-j$) and negative frequencies by $+90^\circ$ (multiply by $j$).
        """)
    
    # col1, col2 = st.columns([1, 2])
    N_taps = st.slider("Filter Length (N)", 11, 101, 31, step=2, help="Must be Odd for Type III/IV filters")
    h = design_hilbert_fir(N_taps)
    
    # with col1:
        # st.subheader("Design Parameters")
        # N_taps = st.slider("Filter Length (N)", 11, 101, 31, step=2, help="Must be Odd for Type III/IV filters")
        # # Design Filter
        # h = design_hilbert_fir(N_taps)
        
    # with col2:
    fig1, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig1.patch.set_alpha(0)
        
    # 1. Impulse Response
    ax[0].stem(np.arange(N_taps), h, basefmt=" ", linefmt='b-', markerfmt='bo')
    ax[0].set_title(f"Impulse Response (N={N_taps})")
    ax[0].set_xlabel("Sample n")
    ax[0].grid(True, alpha=0.3)
        
    # 2. Frequency Response
    w, H = signal.freqz(h, worN=1024)
    w_norm = w / np.pi
        
    # Plot Magnitude and Phase
    ax2 = ax[1].twinx()
        
    ln1 = ax[1].plot(w_norm, 20*np.log10(abs(H)+1e-12), 'b', label='Magnitude (dB)')
    ln2 = ax2.plot(w_norm, np.angle(H), 'g--', label='Phase (rad)')
        
    ax[1].set_xlabel("Normalized Frequency ($\times \pi$)")
    ax[1].set_ylabel("Magnitude (dB)", color='b')
    ax2.set_ylabel("Phase (radians)", color='g')
    ax[1].set_ylim(-50, 5)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax2.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        
    # Add legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc='center right')
        
    ax[1].grid(True, alpha=0.3)
    ax[1].set_title("Frequency Response")
        
    st.pyplot(fig1)
        
    # st.info("""
    # **Observe:**
    # * **Impulse Response:** It is **Anti-symmetric** (Odd symmetry). This is characteristic of Hilbert transformers.
    # * **Phase:** Notice the phase is approximately $\pi/2$ (or $-\pi/2$ depending on delay) in the passband. The ideal shift is exactly 90 degrees.
    # """)
    with st.expander("🔍 Observation"):
        st.markdown(r"""* **Impulse Response:** It is **Anti-symmetric** (Odd symmetry). This is characteristic of Hilbert transformers.
            * **Phase:** Notice the phase is approximately $\pi/2$ (or $-\pi/2$ depending on delay) in the passband. The ideal shift is exactly 90 degrees.
            """)

# ==============================================================================
# TAB 2: ANALYTIC SIGNALS
# ==============================================================================
with tab2:
    # st.header("2. Analytic Signals & Instantaneous Envelope")
    with st.expander("📋 Instructions"):
        st.markdown(r"""
            An **Analytic Signal** $x_a(t)$ is constructed by adding the signal as Real part and its Hilbert Transform as Imaginary part:
            $$ x_a(t) = x(t) + j \cdot \mathcal{H}\{x(t)\} $$
            The magnitude $|x_a(t)|$ gives the **Instantaneous Envelope**.
            """)
    
    # --- Controls (Top Row) ---
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            sig_type = st.selectbox("Signal Type", ["AM Signal", "Chirp"])
        
        # Initialize common time vector
        fs = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(fs*duration), endpoint=False)
        
        # Setup inputs based on type
        if sig_type == "AM Signal":
            with col2:
                fc = st.slider("Carrier Freq (Hz)", 50, 200, 100)
            with col3:
                fm = st.slider("Modulation Freq (Hz)", 1, 20, 5)
            with col4:
                mod_index = st.slider("Modulation Index", 0.0, 1.0, 0.8)
            
            # Calculate AM Signal (after all inputs are gathered)
            envelope = (1 + mod_index * np.cos(2*np.pi*fm*t))
            sig = envelope * np.cos(2*np.pi*fc*t)
            
        else: # Chirp
            with col2:
                f_start = st.slider("Start Freq", 10, 50, 20)
            with col3:
                f_end = st.slider("End Freq", 100, 300, 200)
            # col4 is empty for Chirp
            
            # Calculate Chirp Signal
            sig = signal.chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
            envelope = np.ones_like(sig) # Ideal envelope is 1

    # --- Processing & Plotting ---
    
    # Calculate Analytic Signal
    analytic_sig = hilbert(sig)
    
    extracted_envelope = np.abs(analytic_sig)
    extracted_phase = np.unwrap(np.angle(analytic_sig))
    inst_freq = np.diff(extracted_phase) / (2.0*np.pi) * fs
    
    # --- Plots ---
    fig2, ax = plt.subplots(3, 1, figsize=(10, 9))
    fig2.patch.set_alpha(0)
    
    # 1. Time Domain & Envelope
    ax[0].plot(t, sig, 'k', alpha=0.3, label='Original Signal')
    ax[0].plot(t, extracted_envelope, 'r--', linewidth=2, label='Extracted Envelope (|x_a|)')
    if sig_type == "AM Signal":
        ax[0].plot(t, envelope, 'g:', label='True Envelope')
        
    ax[0].set_title("Instantaneous Amplitude extraction")
    ax[0].legend(loc='upper right')
    ax[0].set_xlim(0, 0.5) # Zoom in
    ax[0].grid(True, alpha=0.3)
    
    # 2. Real vs Imaginary (The 90 deg shift)
    ax[1].plot(t, np.real(analytic_sig), 'b', label='Real (Original)')
    ax[1].plot(t, np.imag(analytic_sig), 'orange', label='Imag (Hilbert)')
    ax[1].set_title("Analytic Components (Real vs Imag)")
    ax[1].legend(loc='upper right')
    ax[1].set_xlim(0, 0.1) # Zoom in more
    ax[1].grid(True, alpha=0.3)
    
    # 3. Instantaneous Frequency
    ax[2].plot(t[1:], inst_freq, 'purple')
    ax[2].set_title("Instantaneous Frequency")
    ax[2].set_ylabel("Frequency (Hz)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(True, alpha=0.3)
    
    # Adjust y-limits for chirp to see the sweep clearly
    if sig_type == "Chirp":
        ax[2].set_ylim(0, max(f_start, f_end) * 1.5)
    
    st.pyplot(fig2)

# ==============================================================================
# TAB 3: SSB MODULATION
# ==============================================================================
with tab3:
    # st.header("3. Single Sideband (SSB) Modulation")
    with st.expander("📋 Instructions"):
        st.markdown(r"""
        Standard AM uses double bandwidth. Using the Hilbert Transform, we can cancel one sideband mathematically:
        * **USB:** $\cos(\omega_c t) m(t) - \sin(\omega_c t) \hat{m}(t)$
        * **LSB:** $\cos(\omega_c t) m(t) + \sin(\omega_c t) \hat{m}(t)$
        """)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            f_carrier_ssb = st.slider("SSB Carrier (Hz)", 1000, 3000, 2000)
        with col2:
            sideband = st.radio("Select Sideband", ["USB (Upper)", "LSB (Lower)"])

    # with col_s1:
    #     f_carrier_ssb = st.slider("SSB Carrier (Hz)", 1000, 3000, 2000)
    #     sideband = st.radio("Select Sideband", ["USB (Upper)", "LSB (Lower)"])
        
    # with col_s2:
    # Generate Message Signal (Bandlimited noise/speech proxy)
    np.random.seed(42)
    # Create a signal with 3 distinct tones for clear visualization
    t_ssb = np.linspace(0, 1, 8000)
    msg = (1.0 * np.sin(2*np.pi*300*t_ssb) + 
             0.5 * np.sin(2*np.pi*600*t_ssb) + 
            0.3 * np.sin(2*np.pi*900*t_ssb))
        
    # 1. Hilbert Transform of Message
    msg_analytic = hilbert(msg)
    msg_hat = np.imag(msg_analytic) # The 90 deg shifted version
        
    # 2. Carrier
    carrier_cos = np.cos(2*np.pi*f_carrier_ssb*t_ssb)
    carrier_sin = np.sin(2*np.pi*f_carrier_ssb*t_ssb)
        
    # 3. SSB Math
    if sideband == "USB (Upper)":
        # USB = m*cos - m_hat*sin
        ssb_sig = msg * carrier_cos - msg_hat * carrier_sin
        color = 'green'
    else:
        # LSB = m*cos + m_hat*sin
        ssb_sig = msg * carrier_cos + msg_hat * carrier_sin
        color = 'red'
            
    # --- Visualization ---
    fig3, ax3 = plt.subplots(2, 1, figsize=(10, 8))
    fig3.patch.set_alpha(0)
        
    # Time Domain
    ax3[0].plot(t_ssb[:200], msg[:200], 'k', label='Message')
    ax3[0].plot(t_ssb[:200], ssb_sig[:200], color=color, alpha=0.7, label=f'SSB Signal ({sideband})')
    ax3[0].set_title("Time Domain")
    ax3[0].legend()
        
    # Frequency Domain (Double Sided to show asymmetry)
    plot_spectrum(ssb_sig, 8000, ax3[1], f"Spectrum of {sideband} Signal", color=color, sides='double')
        
    # Mark Carrier
    ax3[1].axvline(f_carrier_ssb, color='gray', linestyle='--', label='Carrier Freq')
    ax3[1].legend()
        
    st.pyplot(fig3)
        
    # st.success(f"""
    # **Analysis:**
    # Look at the spectrum. The Carrier is at {f_carrier_ssb} Hz.
    # * If **USB** is selected, energy appears **only to the right** of the carrier.
    # * If **LSB** is selected, energy appears **only to the left**.
    # This confirms the Hilbert Transform successfully suppressed half the spectrum!.
    # """)
    st.expander("🔎 Analysis"):
        st.markdown(r"""
        Look at the spectrum. The Carrier is at {f_carrier_ssb} Hz.
        * If **USB** is selected, energy appears **only to the right** of the carrier.
        * If **LSB** is selected, energy appears **only to the left**.
        This confirms the Hilbert Transform successfully suppressed half the spectrum!.
        """)
