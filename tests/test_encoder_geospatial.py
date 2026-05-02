"""
Test suite for Geospatial Encoder.

The Geospatial Encoder converts geographic coordinates into sparse distributed
representations (SDRs). It supports 2D encoding with longitude and latitude,
and optional 3D encoding with altitude. The encoder is designed to preserve
spatial locality so that nearby positions have more overlap than distant ones.

Key Features:
  - Encodes longitude/latitude coordinates into SDRs
  - Optionally includes altitude in the representation
  - Wraps longitude values across the dateline
  - Clamps latitude values at valid Web Mercator bounds
  - Uses speed/timestep to influence locality radius
  - Supports deterministic encoding and approximate decoding

Parameter Validation:
  - xy_scale: controls horizontal spatial scaling
  - z_scale: controls altitude scaling
  - timestep: affects movement/locality radius
  - max_radius: maximum neighborhood radius considered
  - use_altitude: enables 3D coordinate encoding
  - n: total SDR size
  - w: number of active bits
  - dims: coordinate dimensionality (2D or 3D)

Tests validate:
  1. Longitude wrapping produces equivalent encodings across the dateline
  2. Latitude clamping at the poles is stable
  3. Higher speed tends to increase locality radius and overlap
  4. Small spatial changes preserve more overlap than large changes
  5. Altitude changes affect encodings in 3D mode
  6. 2D and 3D encoded values can be approximately decoded
  7. Decoding respects wrapped longitude and clamped latitude bounds
"""

import math

import pytest

from psu_capstone.encoder_layer.coordinate_encoder import CoordinateParameters
from psu_capstone.encoder_layer.geospatial_encoder import GeospatialEncoder, GeospatialParameters


def _build_encoder(
    *,
    use_altitude: bool,
    xy_scale: float = 5.0,
    z_scale: float = 0.5,
    timestep: float = 1.0,
    max_radius: int = 10,
):
    coord_params = CoordinateParameters(
        n=400, w=25, seed=123, max_radius=max_radius, dims=3 if use_altitude else 2
    )
    geo_params = GeospatialParameters(
        xy_scale=xy_scale,
        z_scale=z_scale,
        timestep=timestep,
        max_radius=max_radius,
        use_altitude=use_altitude,
    )
    return GeospatialEncoder(geo_params=geo_params, coord_params=coord_params)


def _overlap(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x and y)


def _active_count(a: list[int]) -> int:
    return sum(1 for x in a if x)


def test_encode_wrap_lon_equivalence_over_dateline():
    enc = _build_encoder(use_altitude=False, xy_scale=5.0)

    # same physical longitude, different representations
    a = enc.encode((1.0, 179.9, 10.0))
    b = enc.encode((1.0, 539.9, 10.0))  # 179.9 + 360
    c = enc.encode((1.0, -180.1, 10.0))  # should wrap close to 179.9

    assert a == b
    assert a == c


def test_encode_clamp_lat_at_poles_is_stable():
    enc = _build_encoder(use_altitude=False, xy_scale=5.0)

    # absurd latitudes should clamp to +/-85.05112878
    a = enc.encode((0.0, 0.0, 9999.0))
    b = enc.encode((0.0, 0.0, 85.05112878))
    c = enc.encode((0.0, 0.0, -9999.0))
    d = enc.encode((0.0, 0.0, -85.05112878))

    assert a == b
    assert c == d


def test_encode_speed_increases_locality_radius_and_tends_to_increase_overlap():
    enc = _build_encoder(use_altitude=False, xy_scale=2.0, timestep=2.0, max_radius=100)

    # two nearby positions
    p1 = (0.0, -77.0365, 38.8977)
    p2 = (0.0, -77.0367, 38.9423)

    slow1 = enc.encode((0.5, p1[1], p1[2]))
    slow2 = enc.encode((0.5, p2[1], p2[2]))

    fast1 = enc.encode((30.0, p1[1], p1[2]))
    fast2 = enc.encode((30.0, p2[1], p2[2]))

    overlap_slow = _overlap(slow1, slow2)
    overlap_fast = _overlap(fast1, fast2)

    assert overlap_fast >= overlap_slow


def test_encode_small_position_change_has_more_overlap_than_large_change():
    enc = _build_encoder(use_altitude=False, xy_scale=10.0, timestep=1.0, max_radius=10)

    base = enc.encode((2.0, -77.0365, 38.8977))
    near = enc.encode((2.0, -77.0366, 38.89775))
    far = enc.encode((2.0, -80.0, 41.0))

    overlap_near = _overlap(base, near)
    overlap_far = _overlap(base, far)

    assert overlap_near > overlap_far


def test_encode_altitude_mode_changes_encoding_when_altitude_changes():
    enc = _build_encoder(use_altitude=True, xy_scale=1.0, timestep=1.0, max_radius=10)

    a = enc.encode((1.0, -77.0365, 38.8977, 10.0))
    b = enc.encode((1.0, -77.0365, 38.8977, 200.0))

    assert a != b
    assert _overlap(a, b) < _active_count(a)


def test_decode_round_trip_3d():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(
        xy_scale=5.0,
        timestep=1.0,
        max_radius=10,
        use_altitude=True,
    )

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (3.0, -77.0365, 38.8977, 15.0)  # speed, lon, lat, alt
    sdr = enc.encode(original)

    decoded_pos, conf = enc.decode(sdr)

    assert decoded_pos is not None
    lon, lat, alt = decoded_pos

    assert math.isclose(lon, original[1], abs_tol=1e-4)
    assert math.isclose(lat, original[2], abs_tol=1e-4)
    assert math.isclose(alt, original[3], abs_tol=1.0)

    assert conf > 0.5


def test_decode_round_trip_2d():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(
        xy_scale=5.0,
        timestep=1.0,
        max_radius=10,
        use_altitude=False,
    )

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (2.0, -122.4194, 37.7749)  # speed, lon, lat
    sdr = enc.encode(original)

    decoded_pos, conf = enc.decode(sdr)

    assert decoded_pos is not None
    lon, lat, alt = decoded_pos

    assert math.isclose(lon, original[1], abs_tol=1e-4)
    assert math.isclose(lat, original[2], abs_tol=1e-4)
    assert alt is None
    assert conf > 0.5


def test_decode_respects_wrap_and_clamp():
    coord_params = CoordinateParameters(n=400, w=25)
    geo_params = GeospatialParameters(xy_scale=5.0, use_altitude=False)

    enc = GeospatialEncoder(geo_params, coord_params)

    original = (2.0, 190.0, 95.0)

    encoded = enc.encode(original)
    decoded, conf = enc.decode(encoded)

    assert decoded is not None
    lon, lat, _ = decoded

    assert -180.0 <= lon < 180.0

    assert -85.05112878 <= lat <= 85.05112878
