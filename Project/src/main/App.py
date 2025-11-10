
import ....SDR_Encoder_Temp. as DateEncoder






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

    enc = DateEncoder(params)

    out = SDR(dimensions=[enc.size])
    enc.encode(time.time(), out)  # current time
    print("Output size:", out.size)
    print("Active indices:", out.getSparse())
