#C:\Users\chris\repos\psu-capstone\Project\src\main\App.py
#C:\Users\chris\repos\psu-capstone\SDR_Encoder_Temp

from __future__ import annotations

from datetime import datetime

from Project.src.main.EncoderLayer.date_encoder import DateEncoder, DateEncoderParameters
from SDR import SDR


if __name__ == "__main__":
    params = DateEncoderParameters(
        season_width=10,
        dayOfWeek_width=5,
        weekend_width=3,
        holiday_width=4,
        timeOfDay_width=6,
        custom_width=3,
        custom_days=["mon,wed,fri"],
        verbose=True,
    )

    date_test = DateEncoder(parameters=params)

    out = SDR(dimensions=[date_test.size])
    date_test.encode(datetime.now(), out)  # current time
    print("Output size:", out.size)
    print("Active indices:", out.getSparse())
