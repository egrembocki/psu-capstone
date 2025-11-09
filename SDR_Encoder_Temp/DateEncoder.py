from dataclasses import dataclass, field
from math import ceil
from time import mktime
from typing import List
import ScalarEncoder
from SDR_Encoder_Temp.BaseEncoder import BaseEncoder
from datetime import datetime, timedelta, time
import time

from SDR_Encoder_Temp.SDR import SDR


@dataclass
class DateEncoderParameters:
    custom_dates: List[str]
    season_width: int = 0
    season_radius: float = 91.5
    dayOfWeek_width: int = 0
    dayOfWeek_radius: float = 1.0
    weekend_width: int = 0
    holiday_width: int = 0
    holiday_dates: List[int] = field(default_factory=lambda: [12, 25])
    timeOfDay_width: int = 0
    timeOfDay_radius: float = 4.0
    custom_width: int = 0
    verbose: bool = False

class DateEncoder(BaseEncoder):
    def __init__(self, parameters: DateEncoderParameters):

        super().__init__()
        self.dayOfWeek_encoder = None
        self.season_encoder = None
        self.holidayEncoder = None
        self.timeOfDayEncoder = None
        self.customDayEncoder = None
        self.weekendEncoder = None

        parameters = self.check_parameters(parameters)

        self.parameters = parameters
        self.custom_dates = parameters.custom_dates
        self.season_width = parameters.season_width
        self.season_radius = parameters.season_radius
        self.dayOfWeek_width = parameters.dayOfWeek_width
        self.dayOfWeek_radius = parameters.dayOfWeek_radius
        self.weekend_width = parameters.weekend_width
        self.holiday_width = parameters.holiday_width
        self.custom_width = parameters.custom_width
        self.holiday_dates = parameters.holiday_dates
        self.timeOfDay_width = parameters.timeOfDay_width
        self.timeOfDay_radius = parameters.timeOfDay_radius
        self.verbose = parameters.verbose

    def encode(self, input_value, output):
        sdrs = []
        season_output: SDR
        day_of_week_output: SDR
        weekend_output: SDR
        customDay_output: SDR
        holiday_output: SDR
        timeOfDay_output: SDR

        print(f"DateEncoder for {input_value.strftime('%a %b %d %H:%M:%S %Y')} {('dst' if input_value.dst() else '')}")

        if self.season_width != 0:
            day_of_year = input_value.timetuple().tm_yday
            season_output = SDR(self.season_encoder.dimensions)
            season_output = self.season_encoder.encode(day_of_year, season_output)
            #I am not sure what the bucket enum is used for here
            print(f" season: {day_of_year} ==> {season_output}")
            sdrs.append(season_output)

        if self.dayOfWeek_width != 0:
            dayofweek = input_value.timetuple().tm_wday + 6 % 7
            day_of_week_output = SDR(self.dayOfWeek_encoder.dimensions)
            day_of_week_output = self.dayOfWeek_encoder.encode(dayofweek, day_of_week_output)
            #another bucket thing here
            print(f" dayOfWeek: {dayofweek} ==> {day_of_week_output}")
            sdrs.append(day_of_week_output)

        if self.weekend_width != 0:
            if input_value.timetuple.tm_wday == 0 or input_value.timetuple.tm_wday == 6 or (input_value.timetuple.tm_wday == 5 and input_value.timetuple.tm_hour > 18):
                val = 1.0
            else:
                val = 0.0
            weekend_output = SDR(self.weekendEncoder.dimensions)
            weekend_output = self.weekendEncoder.encode(val, weekend_output)
            #another bucket thing here
            print(f" weekend: {val} ==> {weekend_output}")
            sdrs.append(weekend_output)

        if self.custom_width != 0:
            customDay = 0.0
            if input_value.timetuple().tm_yday in self.custom_dates:
                customDay = 1.0
            customDay_output = SDR(self.customDayEncoder.dimensions)
            customDay_output = self.customDayEncoder(customDay, customDay_output)
            print(f" customDay: {customDay} ==> {customDay_output}")
            sdrs.append(customDay_output)

        if self.holiday_width != 0:
            val = 0.0
            SECONDS_PER_DAY = 86400
            input = input_value.timetuple()
            for h in self.holiday_dates:
                if len(h) == 3:
                    hdate = #not done




        pass

    def check_parameters(self, parameters: DateEncoderParameters):
        args = parameters
        size = 0
        bucket_map = {}
        buckets = []
        custom_days_set = set()

        #Season Attribute
        if args.season_width != 0:
            p = ScalarEncoder.ScalarEncoderParameters(
                minimum = 0,
                maximum = 366,
                periodic = True,
                activeBits = args.season_width,
                radius = args.season_radius
            )
            self.season_encoder = ScalarEncoder.ScalarEncoder(p)
            print(f"create season encoder: {(p.maximum - p.minimum) / self.season_encoder.radius} buckets, "
                  f"{self.season_encoder.activeBits} bits per bucket, width {self.season_encoder.size}")
            size += self.season_encoder.size
            bucket_map['SEASON'] = len(buckets)
            buckets.append(0.0)

        #Day of week attribute
        if args.dayOfWeek_width != 0:
            p = ScalarEncoder.ScalarEncoderParameters(
                minimum = 0,
                maximum = 7,
                periodic = True,
                activeBits = args.dayOfWeek_width,
                radius = args.dayOfWeek_radius
            )
            self.dayOfWeek_encoder = ScalarEncoder.ScalarEncoder(p)
            print(f"DayOfWeek Encoder: {(p.maximum - p.minimum) / self.dayOfWeek_encoder.radius} categories, "
                  f"{self.dayOfWeek_encoder.activeBits} bits per bucket, width {self.dayOfWeek_encoder.size}")
            size += self.dayOfWeek_encoder.size
            bucket_map['DAYOFWEEK'] = len(buckets)
            buckets.append(0.0)

        #Weekend attribute
        if args.weekend_width != 0:
            p = ScalarEncoder.ScalarEncoderParameters(
                minimum = 0,
                maximum = 1,
                periodic = True,
                activeBits = args.weekend_width
            )
            self.weekendEncoder = ScalarEncoder.ScalarEncoder(p)
            print(f"Weekend Encoder: {(p.maximum - p.minimum) / self.weekendEncoder.radius} categories, "
                  f"{self.weekendEncoder.activeBits} bits per bucket, width {self.weekendEncoder.size}")
            size += self.weekendEncoder.size
            bucket_map['WEEKEND'] = len(buckets)
            buckets.append(0.0)

        #Custom Days attribute
        if args.custom_width != 0:
            custom_dates = args.custom_dates
            daymap = {'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}
            for dayToParse in args.custom_dates:
                dayToParse = dayToParse.lower()
                cust = dayToParse.split(',')
                for day in cust:
                    day = day.strip()
                    if len(day) < 3:
                        raise ValueError(f"DateEncoder: custom; parse error near {day}")
                    day_key = day[:3]
                    if day_key not in daymap:
                        raise ValueError(f"DayEncoder: custom; parse error near {day_key}")
                    custom_dates.append(daymap[day_key])

            p = ScalarEncoder.ScalarEncoderParameters(
                activeBits = args.custom_width,
                minimum = 0,
                maximum = 1,
                category = True,
            )
        self.customDayEncoder = ScalarEncoder.ScalarEncoder(p)
        print(f" create customDays Encoder: boolean, On or Off, {p.activeBits} "
              f"bits per bucket, width {self.customDayEncoder.size}")
        size += self.customDayEncoder.size
        bucket_map['CUSTOM'] = len(buckets)
        buckets.append(0.0)

        #Holiday attribute
        if args.holiday_width != 0:
            for day in args.holiday_dates:
                if len(day) not in [2,3]:
                    raise ValueError(f"DateEncoder: holiday, expecting mon,day or year,mon,day.")

            p = ScalarEncoder.ScalarEncoderParameters(
                activeBits = args.holiday_width,
                minimum = 0,
                maximum = 2,
                periodic = True,
                radius = 1,
            )
            self.holidayEncoder = ScalarEncoder.ScalarEncoder(p)
            print(f" create holiday Encoder: "
                  f"{(p.maximum - p.minimum) / self.holidayEncoder.radius} buckets, "
                  f"{self.holidayEncoder.activeBits} bits per bucket, width {self.holidayEncoder.size}"
                  )
            size += self.holidayEncoder.size
            bucket_map['HOLIDAY'] = len(buckets)
            buckets.append(0.0)

        #Time of day attributes
        if args.timeOfDay_width != 0:
            p = ScalarEncoder.ScalarEncoderParameters(
                minimum = 0,
                maximum = 24,
                periodic = True,
                activeBits = args.timeOfDay_width,
                radius = args.timeOfDay_radius
            )
            self.timeOfDayEncoder = ScalarEncoder.ScalarEncoder(p)
            num_buckets = ceil((self.timeOfDayEncoder.maximum - self.timeOfDayEncoder.minimum) / self.timeOfDayEncoder.radius)
            print(f" create TimeOfDay Encoder:  {num_buckets} buckets, "
                  f"{self.timeOfDayEncoder.activeBits} bits per bucket, width {self.timeOfDayEncoder.size}")
            size += self.timeOfDayEncoder.size

            bucket_map['TIMEOFDAY'] = len(buckets)
            buckets.append(0.0)

        if size <= 0:
            raise ValueError("DateEncoder: no active attributes specified.")
        BaseEncoder.initialize(self, dimensions = [int(size)])
        return args

