
"""
Contextual introduction page.
Informative page about exoplanet detection methods and the challenges of nulling interferometry.
"""

import streamlit as st
from pathlib import Path

# --- Path Setup ---
ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="Context: Exoplanet Detection",
    page_icon="🌌",
    layout="wide",
)

st.title("Context: The Quest for Habitable Worlds 🌌")

st.markdown(r"""
The detection and characterization of exoplanets (planets orbiting stars other than the Sun) is one of the most dynamic fields in modern astrophysics.
Since the discovery of *51 Pegasi b* in 1995, more than 5000 exoplanets have been confirmed.
However, finding a "Twin Earth" (a rocky planet in the habitable zone of a solar-type star) remains a major technological challenge.
""")

st.header("Overview of Detection Methods")

st.markdown("Exoplanet detection relies on various techniques, each sensitive to different planet populations.")

cols = st.columns(2)

with cols[0]:
    img_path = ROOT / "docs" / "img" / "detection_methods.png"
    if img_path.exists():
        st.image(str(img_path), caption="Sensitivity Comparison: Mass vs Orbital Period for different methods.", use_container_width=True)
    else:
        # Fallback text if image missing
        st.info("Sensitivity map image not found.")
        st.markdown("Each method occupies a specific niche depending on planet mass and separation.")

with cols[1]:
    tabs = st.tabs([
        "📉 Radial Velocity", 
        "🌑 Transit", 
        "📏 Astrometry", 
        "🔍 Microlensing", 
        "⏱️ Timing",
        "📸 Direct Imaging", 
    ])

    with tabs[0]:
        st.markdown("""
        **Principle**: Measures the periodic frequency shift (Doppler effect) of the star's light as it orbits the common center of mass with the planet.
        *   **Observable**: "Wobble" of the star along the line of sight.
        *   **Pros**: 
            *   Effective for detecting massive planets (Gas Giants) and close-in planets.
            *   Historically the first successful method for main-sequence stars (51 Peg b).
        *   **Cons**: 
            *   Only provides minimum mass ($M \sin i$) due to unknown inclination.
            *   Stellar activity can mimic planetary signals.
        """)

    with tabs[1]:
        st.markdown("""
        **Principle**: Measures the tiny dip in stellar brightness when a planet passes directly in front of the star (Transit).
        *   **Observable**: Light curve (Flux vs Time).
        *   **Pros**:
            *   Gives the **Radius** of the planet.
            *   Combined with RV, yields density (rocky vs gaseous?).
            *   Allows **Transmission Spectroscopy** to study atmospheric composition (e.g., JWST).
        *   **Cons**:
            *   Geometric probability of transit is low ($R_*/a$).
            *   Requires continuous monitoring of many stars (Kepler, TESS, PLATO).
        """)

    with tabs[2]:
        st.markdown("""
        **Principle**: Detecting variations in the timing of strict periodic events.
        *   **Pulsar Timing**: Timing variations of radio pulses from a neutron star (First confirmed exoplanets, 1992).
        *   **Transit Timing Variations (TTV)**: Gravitational interaction between multiple planets causes transits to happen early or late.
        *   **Pros**: Extremely sensitive to low-mass planets.
        *   **Cons**: Specific to Pulsars or multi-planet transit systems.
        """)

    with tabs[3]:
        st.markdown("""
        **Principle**: Measures the precise positional displacement of the star on the sky due to the planet's gravity.
        *   **Observable**: 2D position ($\alpha, \delta$) over time.
        *   **Pros**:
            *   Best for planets **far** from the star (large orbit = large displacement).
            *   Provides **True Mass** (no $\sin i$ ambiguity).
            *   Major capability of the **Gaia** mission.
        *   **Cons**:
            *   Requires micro-arcsecond precision.
            *   Displacement is tiny and takes years (long orbits) to detect.
        """)

    with tabs[4]:
        st.markdown("""
        **Principle**: Relies on General Relativity. When a foreground star (lens) passes in front of a background star (source), its gravity magnifies the light. A planet acts as a secondary lens, creating a spike in magnification.
        *   **Observable**: Magnification curve over time.
        *   **Pros**:
            *   Most sensitive method for distant, earth-mass (or even free-floating) planets.
            *   Independent of light from the host star.
        *   **Cons**:
            *   **One-time event**: Cannot re-observe or confirm the planet.
            *   Parameter degeneracy.
        """)

    with tabs[5]:
        st.markdown("""
        **Principle**: Capturing the light emitted or reflected by the planet itself by blocking the overwhelming glare of the star.
        *   **Obervable**: Photons from the planet (Image/Spectrum).
        *   **Pros**:
            *   Access to the planet's **Atmosphere** and surface properties without transit.
            *   Best for young, hot, massive planets far from their star.
        *   **Cons**:
            *   **Contrast Challenge**: Star is $10^6$-$10^{10}$ times brighter.
            *   **Resolution Challenge**: Planet is extremely close angularly.
            *   Requires Coronagraphy or Interferometry.
        """)

st.divider()

st.header("Challenges of Direct Imaging")

st.markdown(r"""
To image a habitable Earth around a solar-type star, we must overcome two major physical obstacles:

1.  **Contrast (Flux Ratio)**:
    The star is much brighter than the planet. In the thermal infrared (where the planet emits its own heat), the flux ratio is around **$10^{-6}$ to $10^{-7}$**. In visible light (reflected light), it drops to **$10^{-10}$**.
    *It's like trying to see a firefly sitting on the rim of a blinding lighthouse.*

2.  **Angular Separation (Resolution)**:
    An Earth at 1 AU orbiting a star at 10 parsecs (33 light-years) appears at an angle of **0.1 arcsecond (100 mas)**.
    To distinguish the planet from the star (Rayleigh criterion $\theta = 1.22 \lambda / D$), a traditional (monolithic) telescope would need a gigantic diameter $D$.
    *   At $\lambda = 10 \mu m$ (Thermal Infrared), it would require a telescope of **~25 meters** in diameter.
    *   This is technically and financially colossal for a space telescope.
""")

st.divider()

st.header("The Solution: Nulling Interferometry")

col_nulling_txt, col_nulling_img = st.columns([1, 1])

with col_nulling_txt:
    st.markdown(r"""
    Nulling Interferometry, proposed by Bracewell in 1978, bypasses the telescope size problem by combining light from **several small telescopes** spaced apart.

    **The Principle:**
    1.  Light is collected by multiple telescopes separated by a baseline $B$ (which defines resolution, $D \approx B$).
    2.  Beams are combined by introducing a **$\pi$ (180°) phase shift** between them.
    3.  **On-axis (The Star)**: Waves are in phase opposition and cancel out (Destructive Interference) $\rightarrow$ The star "disappears".
    4.  **Off-axis (The Planet)**: Due to the slightly different arrival angle, waves travel different path lengths. If this angle places the planet on a bright fringe (Constructive Interference), its light is transmitted.

    This achieves the **resolution** of a giant telescope (defined by the telescope spacing) while **extinguishing** the blinding light of the star.
    """)

with col_nulling_img:
    img_path = ROOT / "docs" / "img" / "nulling_principle.png"
    if img_path.exists():
        st.image(str(img_path), caption="Principle of Bracewell's Nulling Interferometry.", use_container_width=True)
    else:
        st.warning(f"Image not found at: {img_path}")

st.divider()

st.header("Challenges of Interferometry")

st.markdown("""
While the principle is elegant, its practical implementation is formidable:

*   **Achromaticity**: Nulling must work over a wide spectral band (to analyze the atmosphere). However, a geometric phase shift depends on wavelength. Complex achromatic phase shifters are required.
*   **Extreme Stability**: Optical paths must be controlled with nanometric precision ($\lambda/1000$). The slightest vibration destroys the star extinction (*nulling depth*).
*   **Sensitivity**: Splitting light among multiple telescopes reduces the signal per channel.
*   **Ambiguity**: Interference fringes create a striped sensitivity map (Transmission Map) which makes exact planet localization difficult without modulation (array rotation, phase modulation).

This is the context for ambitious missions like **LIFE (Large Interferometer For Exoplanets)**, aiming to deploy such an array in space.
""")
