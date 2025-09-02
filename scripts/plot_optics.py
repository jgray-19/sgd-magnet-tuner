from matplotlib import pyplot as plt

from aba_optimiser.config import SEQUENCE_FILE
from aba_optimiser.mad.mad_interface import MadInterface

mad = MadInterface(
    sequence_file=SEQUENCE_FILE, magnet_range="$start/$end", bpm_pattern="BPM"
)
tws = mad.run_twiss()

plt.plot(tws["s"], tws["beta11"], label="Beta X")
plt.plot(tws["s"], tws["beta22"], label="Beta Y")
plt.xlabel("Position (m)")
plt.ylabel("Beta function (m)")
plt.legend()
plt.title("LHC Beta Functions")
plt.grid()
plt.show()
