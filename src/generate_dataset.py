"""Physically-informed synthetic dataset generator for oil probability prediction.

Simulates hydrocarbon microseepage: methane concentration and pressure
anomalies that decay exponentially with distance from known oil field
anchors, following the geological principle that petroleum reservoirs leak
light hydrocarbons to the surface through imperfect seals. See CLAUDE.md
section 4 for the full specification and scientific references.

The exact decay scales and noise levels are didactic simplifications
calibrated so the resulting classification problem is learnable by a
simple model (see the "critical attention point" in CLAUDE.md section 4.4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Oil field and basin anchors along the Brazilian coast, from SE to NE.
# Coordinates are approximate centroids of producing fields or basin depocenters.
# Used illustratively to generate a learnable signal, not for real exploration.
OIL_FIELD_ANCHORS = {
    # Bacia de Santos (pre-sal, SE)
    "Tupi/Lula":  (-25.20, -43.00),
    "Buzios":     (-25.30, -43.50),
    "Sapinhoa":   (-25.00, -43.30),
    # Bacia de Campos (SE)
    "Marlim":     (-22.50, -40.30),
    "Roncador":   (-22.05, -40.10),
    "Barracuda":  (-22.35, -40.10),
    "Albacora":   (-22.20, -40.30),
    # Bacia do Espírito Santo
    "Jubarte":    (-20.60, -39.80),
    "Camarupim":  (-19.80, -39.50),
    # Bacia Camamu-Almada (BA)
    "Camamu":     (-13.80, -38.80),
    # Bacia Sergipe-Alagoas (SE/AL)
    "Sergipe_Alagoas": (-10.50, -36.50),
    # Bacia Potiguar (RN) — maior província petrolífera do NE
    "Potiguar_Offshore": (-4.00, -36.80),
    "Potiguar_Onshore":  (-5.10, -36.50),
    # Bacia do Ceará (CE)
    "Ceara":      (-3.20, -38.50),
    # Bacia de Barreirinhas (MA/PI) — próxima à Baía de São Luís
    "Barreirinhas": (-2.50, -42.50),
}

# Sampling bounding box covering the full Brazilian continental margin.
# Expanded from the original Campos/Santos box to include NE basins.
LAT_RANGE = (-34.0, 5.5)
LON_RANGE = (-50.0, -28.0)

# Background atmospheric levels.
METHANE_BACKGROUND_PPM = 1.9
PRESSURE_BACKGROUND_HPA = 1013.0

# Anomaly amplitude and spatial decay scale (km).
# Decay scales were recalibrated from 35/45 km (Campos/Santos local model) to
# 80/100 km (coast-wide basin model) after expanding the bounding box to the full
# Brazilian margin. At 35 km scale, > 95% of points in the larger bbox fall in the
# near-zero signal region, collapsing prevalence to < 1% and making the problem
# unlearnable. Larger scales represent diffuse basin-level leakage rather than
# point-seep anomalies — a known distinction in the microseepage literature
# (Saunders et al., 1999). These remain didactic simplifications.
METHANE_ANOMALY_AMPLITUDE_PPM = 0.35
METHANE_DECAY_SCALE_KM = 80.0
PRESSURE_ANOMALY_AMPLITUDE_HPA = 4.0
PRESSURE_DECAY_SCALE_KM = 100.0

# Measurement noise.
METHANE_NOISE_STD_PPM = 0.03
PRESSURE_NOISE_STD_HPA = 0.5

# Label-generating logit: methane is the primary signal, pressure secondary.
METHANE_LOGIT_WEIGHT = 7.0
PRESSURE_LOGIT_WEIGHT = 3.0
LOGIT_BIAS = -4.5
LOGIT_NOISE_STD = 0.25

N_SAMPLES = 30_000
RANDOM_STATE = 42

EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Great-circle distance in km between arrays of points and a fixed point."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_to_nearest_field_km(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    distances = np.stack(
        [
            haversine_distance_km(latitude, longitude, lat, lon)
            for lat, lon in OIL_FIELD_ANCHORS.values()
        ],
        axis=1,
    )
    return distances.min(axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def biogenic_methane_anomaly(
    latitude: np.ndarray, longitude: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Non-petroleum methane from wetlands, agriculture, and urban sources.

    Each region is based on documented CH4 emission sources in the Brazilian
    NE coast — in particular Maranhão (MA), where the OBSAT will take place.
    These contributions are ADDED to atmospheric background but are NOT
    correlated with oil presence, so the model must learn to use geographic
    location to distinguish oil-seepage anomalies from biogenic interference.

    Sources and scientific basis:
    - Baixada Maranhense (wetlands + rice): Saunois et al. (2020); Ramsar site MA-001.
      One of Brazil's largest seasonal floodplain systems (~13,000 km²). Anaerobic
      decomposition in flooded soils and submerged rice paddies are significant CH4
      emitters (Yan et al., 2009).
    - Delta do Parnaíba (MA/PI coastal wetlands): Melack & Hess (2010). Extensive
      tidal wetlands and estuaries generate biogenic methane year-round.
    - Amazon/Cerrado transition (northern MA, PA): Saunois et al. (2020). Tropical
      wetlands are the single largest natural CH4 source globally (~30% of emissions).
    - Cerrado interior (cattle ranching, soy, fires): MCTI/SEEG Brazil inventories.
      Enteric fermentation from MA's large cattle herd and land-use change fires
      contribute measurable CH4 locally.
    - São Luís urban core: IPCC (2006) Tier 1 estimates. Landfills, wastewater
      treatment, and industrial fugitive emissions elevate urban CH4.
    - Global low-level anthropogenic baseline: diffuse contribution everywhere
      (fossil fuel infrastructure, biomass burning transport).
    """
    anomaly = np.zeros(len(latitude))

    # Baixada Maranhense — seasonal floodplain + rice paddies (western MA)
    mask = (
        (latitude >= -5.0) & (latitude <= -2.0) &
        (longitude >= -46.5) & (longitude <= -43.5)
    )
    anomaly[mask] += rng.uniform(0.04, 0.20, int(mask.sum()))

    # Delta do Parnaíba — coastal wetlands at MA/PI border
    mask = (
        (latitude >= -3.0) & (latitude <= -2.4) &
        (longitude >= -42.8) & (longitude <= -41.5)
    )
    anomaly[mask] += rng.uniform(0.03, 0.13, int(mask.sum()))

    # Amazon / northern Cerrado transition (northern part of bbox: north MA, PA, AP)
    mask = latitude > -1.0
    anomaly[mask] += rng.uniform(0.05, 0.25, int(mask.sum()))

    # Cerrado interior — cattle ranching and agricultural fires (central/south MA)
    mask = (
        (latitude >= -9.0) & (latitude <= -4.0) &
        (longitude >= -47.0) & (longitude <= -43.0)
    )
    anomaly[mask] += rng.uniform(0.02, 0.10, int(mask.sum()))

    # São Luís urban core — landfills, wastewater, industrial fugitives
    mask = (
        (latitude >= -2.75) & (latitude <= -2.35) &
        (longitude >= -44.45) & (longitude <= -44.05)
    )
    anomaly[mask] += rng.uniform(0.02, 0.08, int(mask.sum()))

    # Diffuse global anthropogenic baseline (fossil fuel infrastructure, biomass
    # burning transport) — small but present everywhere
    anomaly += rng.exponential(0.015, len(latitude))

    return anomaly


def generate_dataset(
    n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    latitude = rng.uniform(LAT_RANGE[0], LAT_RANGE[1], n_samples)
    longitude = rng.uniform(LON_RANGE[0], LON_RANGE[1], n_samples)
    distance_km = distance_to_nearest_field_km(latitude, longitude)

    methane_anomaly = METHANE_ANOMALY_AMPLITUDE_PPM * np.exp(
        -distance_km / METHANE_DECAY_SCALE_KM
    )
    non_oil_methane = biogenic_methane_anomaly(latitude, longitude, rng)
    methane_ppm = (
        METHANE_BACKGROUND_PPM
        + methane_anomaly
        + non_oil_methane
        + rng.normal(0, METHANE_NOISE_STD_PPM, n_samples)
    )

    pressure_anomaly = PRESSURE_ANOMALY_AMPLITUDE_HPA * np.exp(
        -distance_km / PRESSURE_DECAY_SCALE_KM
    )
    pressure_hpa = (
        PRESSURE_BACKGROUND_HPA
        + pressure_anomaly
        + rng.normal(0, PRESSURE_NOISE_STD_HPA, n_samples)
    )

    methane_z = methane_anomaly / METHANE_ANOMALY_AMPLITUDE_PPM
    pressure_z = pressure_anomaly / PRESSURE_ANOMALY_AMPLITUDE_HPA
    logit = (
        METHANE_LOGIT_WEIGHT * methane_z
        + PRESSURE_LOGIT_WEIGHT * pressure_z
        + LOGIT_BIAS
        + rng.normal(0, LOGIT_NOISE_STD, n_samples)
    )
    true_probability = sigmoid(logit)
    label = rng.binomial(1, true_probability)

    return pd.DataFrame(
        {
            "latitude": latitude,
            "longitude": longitude,
            "methane_ppm": methane_ppm,
            "pressure_hpa": pressure_hpa,
            "distance_to_nearest_field_km": distance_km,
            "non_oil_methane_contribution": non_oil_methane,  # analysis only
            "true_probability": true_probability,
            "label": label,
        }
    )


def main() -> None:
    df = generate_dataset()

    output_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} samples -> {output_path}")
    print(f"Positive label prevalence: {df['label'].mean():.4%}")
    print(f"methane_ppm: mean={df['methane_ppm'].mean():.4f}, std={df['methane_ppm'].std():.4f}")
    print(f"pressure_hpa: mean={df['pressure_hpa'].mean():.4f}, std={df['pressure_hpa'].std():.4f}")
    print(f"non_oil_methane: mean={df['non_oil_methane_contribution'].mean():.4f}, "
          f"max={df['non_oil_methane_contribution'].max():.4f}")


if __name__ == "__main__":
    main()
