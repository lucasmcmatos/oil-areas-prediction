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

# Real oil field anchors in the Campos and Santos basins (Brazilian SE coast).
# Used illustratively to generate a learnable signal, not for real exploration.
OIL_FIELD_ANCHORS = {
    "Tupi/Lula": (-25.20, -43.00),
    "Buzios": (-25.30, -43.50),
    "Sapinhoa": (-25.00, -43.30),
    "Marlim": (-22.50, -40.30),
    "Roncador": (-22.05, -40.10),
    "Barracuda": (-22.35, -40.10),
    "Jubarte": (-20.60, -39.80),
    "Albacora": (-22.20, -40.30),
}

# Sampling bounding box covering the Campos/Santos basins and nearby margin.
LAT_RANGE = (-27.5, -19.0)
LON_RANGE = (-45.5, -37.5)

# Background atmospheric levels.
METHANE_BACKGROUND_PPM = 1.9
PRESSURE_BACKGROUND_HPA = 1013.0

# Anomaly amplitude and spatial decay scale (km).
METHANE_ANOMALY_AMPLITUDE_PPM = 0.35
METHANE_DECAY_SCALE_KM = 35.0
PRESSURE_ANOMALY_AMPLITUDE_HPA = 4.0
PRESSURE_DECAY_SCALE_KM = 45.0

# Measurement noise.
METHANE_NOISE_STD_PPM = 0.03
PRESSURE_NOISE_STD_HPA = 0.5

# Label-generating logit: methane is the primary signal, pressure secondary.
METHANE_LOGIT_WEIGHT = 7.0
PRESSURE_LOGIT_WEIGHT = 3.0
LOGIT_BIAS = -5.5
LOGIT_NOISE_STD = 0.25

N_SAMPLES = 12_000
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
    methane_ppm = (
        METHANE_BACKGROUND_PPM
        + methane_anomaly
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


if __name__ == "__main__":
    main()
