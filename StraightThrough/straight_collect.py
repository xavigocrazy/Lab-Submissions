import ugradio
import pandas as pd
from rtlsdr import RtlSdr
import asyncio
import time
import numpy as np

sdr = ugradio.sdr.SDR(sample_rate = 3e6)
data = sdr.capture_data(2048, nblocks = 10)
print(sdr)
print(data)
np.savez("comb_220_200_3e6.npz",data)
