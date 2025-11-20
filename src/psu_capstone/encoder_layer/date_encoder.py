"""Date encoder 2.0 for HTM, ported to Python from the C++ implementation.

This module provides a DateEncoder class that encodes various temporal features
(season, day of week, weekend, custom days, holiday, time of day) into a Sparse
Distributed Representation (SDR) for use in Hierarchical Temporal Memory (HTM) systems.

The encoder uses ScalarEncoder instances for each enabled feature, concatenating
their outputs into a single SDR. The configuration is controlled via the
DateEncoderParameters dataclass.

Usage:
    params = DateEncoderParameters(
        season_width=10,
        day_of_week_width=5,
        weekend_width=3,
        holiday_width=4,
        time_of_day_width=6,
        custom_width=3,
        custom_days=["mon,wed,fri"],
        verbose=True,
    )
    encoder = DateEncoder(params)
    output = SDR(dimensions=[encoder.size])
    encoder.encode(datetime.now(), output)
    print("Output size:", output.size)
    print("Active indices:", output.get_sparse())
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set

import pandas as pd

from psu_capstone.encoder_layer.base_encoder import BaseEncoder
from psu_capstone.encoder_layer.rdse import RandomDistributedScalarEncoder, RDSEParameters
from psu_capstone.encoder_layer.scalar_encoder import ScalarEncoder, ScalarEncoderParameters
from psu_capstone.encoder_layer.sdr import SDR

SIZE_FACTOR = 2


@dataclass
class DateEncoderParameters:
    """Configuration parameters for DateEncoder.

       Each field controls the encoding of a specific temporal feature.
       Set the corresponding width to a nonzero value to enable encoding for that feature.

       Attributes:
           season_width: Number of active bits for season (day of year).
           season_radius: Radius for season encoding (days).
           day_of_week_width: Number of active bits for day of week.
           day_of_week_radius: Radius for day of week encoding.
           weekend_width: Number of active bits for weekend flag.
           holiday_width: Number of active bits for holiday encoding.
           holiday_dates: List of holidays as [month, day] or [year, month, day].
           time_of_day_width: Number of active bits for time of day.
           time_of_day_radius: Radius for time of day encoding (hours).
           custom_width: Number of active bits for custom day groups.
           custom_days: List of custom day group strings (e.g., ["mon,wed,fri"]).


           /**
    * The DateEncoderParameters structure is used to pass configuration parameters to
    * the DateEncoder. These Six (6) members define the total number of bits in the output.
    *     Members:  season, dayOfWeek, weekend, holiday, timeOfDay, customDays
    *
    * Each member is a separate attribute of a date/time that can be activated
    * by providing a width parameter and sometimes a radius parameter.
    * Each is implemented separately using a ScalarEncoder and the results
    * are concatinated together.
    *
    * The width attribute determines the number of bits to be used for each member.
    * and 0 means don't use.  The width is like a weighting to indicate the relitive importance
    * of this member to the overall data value.
    *
    * The radius attribute indicates the size of the bucket; the quantization size.
    * All values in the same bucket generate the same pattern.
    *
    * To avoid problems with leap year, consider a year to have 366 days.
    * The timestamp will be converted to components such as time and dst based on
    * local timezone and location (see localtime()).
    *
    */
    """

    # Season: day of year (0..366), default radius 91.5 days (~4 seasons)
    season_width: int = 0
    """Number of active bits for season (day of year). how many bits to apply to season
      /**
   *  Member: season -  The portion of the year. Unit is day.  Default radius is
   *                    91.5 days which gives 4 seasons per year.
   */
   """

    season_radius: float = 91.5
    """Radius for season encoding, in days (default ~4 seasons).
    days per season
    """

    # Day of week: Monday=0, Tuesday=1, ... (C++ maps from tm_wday)
    day_of_week_width: int = 0
    """Number of active bits for day of week, how many bits to apply to day of week.
    
    """

    day_of_week_radius: float = 1.0
    """Radius for day of week encoding, every day is a separate bucket."""

    # Weekend flag (0/1, Fri 6pm through Sun midnight)
    weekend_width: int = 0
    """Number of active bits for weekend flag."""

    # Holiday: boolean-ish with ramp, default dates = [[12, 25]] (month, day)
    holiday_width: int = 0
    """Number of active bits for holiday encoding."""

    holiday_dates: List[List[int]] = field(default_factory=lambda: [[12, 25]])
    """List of holidays as [month, day] or [year, month, day]."""

    # Time of day: 0..24 hours
    time_of_day_width: int = 0
    """Number of active bits for time of day."""

    time_of_day_radius: float = 4.0
    """Radius for time of day encoding, in hours."""

    # Custom day groups (e.g. ["mon,wed,fri"])
    custom_width: int = 0
    """Number of active bits for custom day groups."""

    custom_days: List[str] = field(default_factory=list)
    """List of custom day group strings (e.g., ["mon,wed,fri"])."""

    rdse_used: bool = True
    """Enable RDSE usage for date encoder."""


class DateEncoder(BaseEncoder):
    """
    Encodes date/time information into a Sparse Distributed Representation (SDR)
    for use in HTM systems. Supports up to six temporal features:

      - season       (day-of-year)
      - dayOfWeek
      - weekend
      - customDays
      - holiday
      - timeOfDay

    Each feature is encoded using a RDSE Encoder, and the resulting SDRs are
    concatenated into a single output SDR.

    Attributes:
        _parameters: Configuration parameters for the encoder.
        _seasonEncoder: RDSE Encoder for season (day of year).
        _dayOfWeekEncoder: RDSE Encoder for day of week.
        _weekendEncoder: RDSE Encoder for weekend flag.
        _customDaysEncoder: RDSE Encoder for custom day groups.
        _holidayEncoder: RDSE Encoder for holidays.
        _timeOfDayEncoder: RDSE Encoder for time of day.
        _customDays: Set of integer day indices for custom days.
        _bucketMap: Mapping from feature index to bucket position.
        _buckets: List of bucket values for each feature.
        _size: Total number of bits in the output SDR.
        _rdse_used: Flag indicating if RDSE is used.
    """

    # Constants for bucketMap keys
    SEASON = 0
    """Index for season feature in bucketMap."""

    DAYOFWEEK = 1
    """Index for day of week feature in bucketMap."""

    WEEKEND = 2
    """Index for weekend feature in bucketMap."""

    CUSTOM = 3
    """Index for custom days feature in bucketMap."""

    HOLIDAY = 4
    """Index for holiday feature in bucketMap."""

    TIMEOFDAY = 5
    """Index for time of day feature in bucketMap."""

    def __init__(
        self, parameters: DateEncoderParameters, dimensions: List[int] | None = None
    ) -> None:
        """
        Initialize the DateEncoder with the given parameters.

        Args:
            parameters: DateEncoderParameters instance specifying encoding options.
            dimensions: Optional SDR dimensions (unused, for compatibility).

        Raises:
            ValueError: If custom_days is specified but empty, or if no widths are provided.
        """
        self._parameters = copy.deepcopy(parameters)
        """DateEncoderParameters: Configuration parameters for the encoder."""
        self._customDays: Set[int] = set()
        """Set of integer day indices for custom days."""
        self._bucketMap: Dict[int, int] = {}
        """Mapping from feature index to bucket position."""
        self._buckets: List[float] = []
        """List of bucket values for each feature."""
        self._size: int = 0
        """Total number of bits DateEncoder."""
        self._rdse_used = parameters.rdse_used
        """Flag indicating if RDSE is used."""

        # Declare one encoder per feature
        self._season_encoder = None
        """Encoder for season (day of year)."""
        self._dayofweek_encoder = None
        """Encoder for day of week."""
        self._weekend_encoder = None
        """Encoder for weekend flag."""
        self._customdays_encoder = None
        """Encoder for custom day groups."""
        self._holiday_encoder = None
        """Encoder for holidays."""
        self._timeofday_encoder = None
        """Encoder for time of day."""

        self._initialize(self._parameters)

        super().__init__(dimensions, self._size)

    @property
    def buckets(self) -> List[float]:
        """Mapping from feature index to bucket position."""
        return self._buckets

    def _initialize(self, parameters: DateEncoderParameters) -> None:
        """
        Configure scalar or RDSE encoders for each enabled feature based on parameters.

        Args:
            parameters: DateEncoderParameters instance.

        Raises:
            ValueError: If custom_days are invalid or no widths are provided.
        """

        size = 0
        self._bucketMap.clear()
        self._buckets.clear()

        use_rdse = self._rdse_used

        # -------- Season feature Encoder --------
        if parameters.season_width != 0:
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.season_width * SIZE_FACTOR,
                    sparsity=0.10,
                    radius=parameters.season_radius,
                )
                self._season_encoder = RandomDistributedScalarEncoder(p)
                assert self._season_encoder is not None
                assert self._season_encoder.size > 0
                encoder_size = self._season_encoder.size
                print("Using RDSE for season encoder", encoder_size)
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=366.0,
                    periodic=True,
                    active_bits=parameters.season_width,
                    radius=parameters.season_radius,
                    size=parameters.season_width * SIZE_FACTOR,
                )
                self._season_encoder = ScalarEncoder(p)
                encoder_size = self._season_encoder.size
            self._bucketMap[self.SEASON] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Day of week --------
        if parameters.day_of_week_width != 0:
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.day_of_week_width * SIZE_FACTOR,
                    sparsity=0.10,
                    radius=parameters.day_of_week_radius,
                )
                self._dayofweek_encoder = RandomDistributedScalarEncoder(p)
                encoder_size = self._dayofweek_encoder.size
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=7.0,
                    periodic=True,
                    active_bits=parameters.day_of_week_width,
                    radius=parameters.day_of_week_radius,
                    size=parameters.day_of_week_width * SIZE_FACTOR,
                )
                self._dayofweek_encoder = ScalarEncoder(p)
                encoder_size = self._dayofweek_encoder.size
            self._bucketMap[self.DAYOFWEEK] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Weekend --------
        if parameters.weekend_width != 0:
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.weekend_width * SIZE_FACTOR, sparsity=0.10, category=True
                )
                self._weekend_encoder = RandomDistributedScalarEncoder(p)
                encoder_size = self._weekend_encoder.size
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=1.0,
                    category=True,
                    active_bits=parameters.weekend_width,
                    size=parameters.weekend_width * SIZE_FACTOR,
                )
                self._weekend_encoder = ScalarEncoder(p)
                encoder_size = self._weekend_encoder.size
            self._bucketMap[self.WEEKEND] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Custom days --------
        if parameters.custom_width != 0:
            if not parameters.custom_days:
                raise ValueError(
                    "DateEncoder: custom_days must contain at least one pattern string."
                )
            daymap = {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6,
            }
            for spec in parameters.custom_days:
                s = spec.lower()
                parts = [x.strip() for x in s.split(",") if x.strip()]
                for day in parts:
                    if len(day) < 3:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    key = day[:3]
                    if key not in daymap:
                        raise ValueError(f"DateEncoder custom_days parse error near '{day}'")
                    self._customDays.add(daymap[key])
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.custom_width * SIZE_FACTOR, sparsity=0.10, category=True
                )
                self._customdays_encoder = RandomDistributedScalarEncoder(p)
                encoder_size = self._customdays_encoder.size
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=1.0,
                    category=True,
                    active_bits=parameters.custom_width,
                    size=parameters.custom_width * SIZE_FACTOR,
                )
                self._customdays_encoder = ScalarEncoder(p)
                encoder_size = self._customdays_encoder.size
            self._bucketMap[self.CUSTOM] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Holiday --------
        if parameters.holiday_width != 0:
            for day in parameters.holiday_dates:
                if len(day) not in (2, 3):
                    raise ValueError(
                        "DateEncoder: holiday_dates entries must be [mon,day] or [year,mon,day]."
                    )
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.holiday_width * SIZE_FACTOR, sparsity=0.10, radius=1.0
                )
                self._holiday_encoder = RandomDistributedScalarEncoder(p)
                encoder_size = self._holiday_encoder.size
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=2.0,
                    periodic=True,
                    active_bits=parameters.holiday_width,
                    size=parameters.holiday_width * SIZE_FACTOR,
                    radius=1.0,
                )
                self._holiday_encoder = ScalarEncoder(p)
                encoder_size = self._holiday_encoder.size
            self._bucketMap[self.HOLIDAY] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Time of day --------
        if parameters.time_of_day_width != 0:
            if use_rdse:
                p = RDSEParameters(
                    size=parameters.time_of_day_width * SIZE_FACTOR,
                    sparsity=0.10,
                    radius=parameters.time_of_day_radius,
                )
                self._timeofday_encoder = RandomDistributedScalarEncoder(p)
                encoder_size = self._timeofday_encoder.size
            else:
                p = ScalarEncoderParameters(
                    minimum=0.0,
                    maximum=24.0,
                    periodic=True,
                    active_bits=parameters.time_of_day_width,
                    size=parameters.time_of_day_width * SIZE_FACTOR,
                    radius=parameters.time_of_day_radius,
                )
                self._timeofday_encoder = ScalarEncoder(p)
                encoder_size = self._timeofday_encoder.size
            self._bucketMap[self.TIMEOFDAY] = len(self._buckets)
            self._buckets.append(0.0)
            size += encoder_size

        # -------- Final checks --------

        assert size > 0, "DateEncoder: At least one width parameter must be nonzero."

        self._size = size  # Total size of the DateEncoder

    def encode(
        self, input_value: datetime | pd.Timestamp | time.struct_time | float, output_sdr: SDR
    ) -> None:
        """
        Encode a timestamp-like value into the provided SDR.

          /**
            * encode the input and generate the output pattern.
            * The input is unix time, same as would be generated by time(0)
            * which is seconds since EPOCH, Jan 1, 1970.
            * Inputs of time_point or struc tm are converted to time_t.
            *
            * Output is an array of 0's and 1's in an SDR container.
            */

        Args:
            input_value: The value to encode. Can be None (current time), int/float (epoch seconds),
                         datetime, or time.struct_time.
            output: SDR instance to write the encoding into.

        Raises:
            ValueError: If output SDR size does not match encoder size.
            TypeError: If input_value type is unsupported.
            RuntimeError: If encoder is misconfigured.
        """
        assert output_sdr.size == self.size, "Output SDR size does not match encoder size."

        if input_value is None:
            t = time.localtime()
        elif isinstance(input_value, (int, float)):
            t = time.localtime(float(input_value))
        elif isinstance(input_value, datetime):
            ts = input_value.timestamp()
            t = time.localtime(ts)
        elif isinstance(input_value, time.struct_time):
            t = input_value
        else:
            raise TypeError(f"Unsupported type for DateEncoder.encode: {type(input_value)}")

        sdrs: List[SDR] = []

        # --- Season: day of year (0-based) ---
        if self._season_encoder is not None:
            day_of_year = float(t.tm_yday - 1)
            encoder = self._season_encoder
            sdr = SDR([encoder.size])
            encoder.encode(day_of_year, sdr)
            radius = encoder._radius

            if radius > 0:
                bucket_idx = math.floor(day_of_year / radius)
                self._buckets[self._bucketMap[self.SEASON]] = float(bucket_idx)

            else:
                self._buckets[self._bucketMap[self.SEASON]] = day_of_year

            sdrs.append(sdr)

        # --- Day of week ---
        if self._dayofweek_encoder is not None:
            c_tm_wday = (t.tm_wday + 1) % 7
            day_of_week = float((c_tm_wday + 6) % 7)
            encoder = self._dayofweek_encoder
            sdr = SDR(encoder.dimensions)
            encoder.encode(day_of_week, sdr)
            radius = encoder._radius

            if radius <= 0:

                bucket_val = day_of_week - math.fmod(day_of_week, radius)
                self._buckets[self._bucketMap[self.DAYOFWEEK]] = bucket_val
            else:
                bucket_idx = math.floor(day_of_week / radius)
                self._buckets[self._bucketMap[self.DAYOFWEEK]] = float(bucket_idx)

            sdrs.append(sdr)
        else:
            c_tm_wday = (t.tm_wday + 1) % 7

        # --- Weekend flag ---
        if self._weekend_encoder is not None:
            val = (
                1.0
                if c_tm_wday == 0 or c_tm_wday == 6 or (c_tm_wday == 5 and t.tm_hour > 18)
                else 0.0
            )
            encoder = self._weekend_encoder
            sdr = SDR(encoder.dimensions)
            encoder.encode(val, sdr)
            self._buckets[self._bucketMap[self.WEEKEND]] = val

            sdrs.append(sdr)

        # --- Custom days ---
        if self._customdays_encoder is not None:
            custom_val = 1.0 if t.tm_wday in self._customDays else 0.0
            encoder = self._customdays_encoder
            sdr = SDR(encoder.dimensions)
            encoder.encode(custom_val, sdr)
            self._buckets[self._bucketMap[self.CUSTOM]] = custom_val

            sdrs.append(sdr)

        # --- Holiday ramp ---
        if self._holiday_encoder is not None:
            val = self._holiday_value(t)
            encoder = self._holiday_encoder
            sdr = SDR(encoder.dimensions)
            encoder.encode(val, sdr)
            self._buckets[self._bucketMap[self.HOLIDAY]] = math.floor(val)

            sdrs.append(sdr)

        # --- Time of day ---
        if self._timeofday_encoder is not None:
            tod = t.tm_hour + t.tm_min / 60.0 + t.tm_sec / 3600.0
            encoder = self._timeofday_encoder
            sdr = SDR(encoder.dimensions)
            encoder.encode(tod, sdr)
            radius = max(getattr(encoder, "_radius", 1e-9), 1e-9)
            bucket_val = tod - math.fmod(tod, radius)
            self._buckets[self._bucketMap[self.TIMEOFDAY]] = bucket_val

            sdrs.append(sdr)

        if not sdrs:
            raise RuntimeError("DateEncoder misconfigured: no sub-encoders enabled.")

        all_sparse: List[int] = []
        offset = 0
        for sdr in sdrs:
            for idx in sdr.get_sparse():
                assert (
                    0 <= idx < output_sdr.size
                ), f"Index {idx} out of bounds for SDR size {output_sdr.size}"
                all_sparse.append(idx + offset)
            offset += sdr.size

        print("output_sdr.size:", output_sdr.size)
        print("sum of sdr sizes:", sum(s.size for s in sdrs))
        print("all_sparse:", all_sparse)

        output_sdr.zero()
        output_sdr.set_sparse(all_sparse)

    # ------------------------------------------------------------------ #
    # Holiday helper (matches C++ logic)
    # ------------------------------------------------------------------ #

    def _holiday_value(self, t: time.struct_time) -> float:
        """
        Compute the holiday ramp value for the given timestamp.

        Args:
            t: time.struct_time representing the date/time.

        Returns:
            float: Holiday ramp value (0.0 if not near a holiday, up to 2.0 if near).
        """
        seconds_per_day = 86400.0
        input_ts = time.mktime(t)

        for h in self._parameters.holiday_dates:
            if len(h) == 3:
                year, mon, day = h
            else:
                year = t.tm_year
                mon, day = h
            h_ts = self.mktime(year, mon, day)

            if input_ts > h_ts:
                diff = input_ts - h_ts
                if diff < seconds_per_day:
                    return 1.0
                elif diff < 2.0 * seconds_per_day:
                    return 1.0 + (diff - seconds_per_day) / seconds_per_day
            else:
                diff = h_ts - input_ts
                if diff < seconds_per_day:
                    return 1.0 - diff / seconds_per_day

        return 0.0

    @staticmethod
    def mktime(year: int, mon: int, day: int, hr: int = 0, minute: int = 0, sec: int = 0) -> float:
        """
        Convert a date/time to Unix epoch seconds.

        Args:
            year: Year (e.g., 2024).
            mon: Month (1-12).
            day: Day (1-31).
            hr: Hour (0-23).
            minute: Minute (0-59).
            sec: Second (0-59).

        Returns:
            float: Seconds since Unix epoch.
        """
        dt = datetime(year, mon, day, hr, minute, sec)
        return time.mktime(dt.timetuple())


if __name__ == "__main__":
    """
    Example usage of DateEncoder. Prints the encoded SDR for the current time.
    """
    params = DateEncoderParameters(
        season_width=10,
        day_of_week_width=5,
        weekend_width=3,
        holiday_width=4,
        time_of_day_width=6,
        custom_width=3,
        custom_days=["mon,wed,fri"],
    )
    encoder = DateEncoder(params)

    print("DateEncoder size:", encoder.size)

    output = SDR([encoder.size])
    encoder.encode(datetime.now(), output)
    print("Output size:", output.size)
    print("Active indices:", output.get_sparse())
