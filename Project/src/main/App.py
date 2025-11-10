
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from SDR_Encoder_Temp.SDR import SDR
from SDR_Encoder_Temp.date_encoder import DateEncoder, DateEncoderParameters


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
