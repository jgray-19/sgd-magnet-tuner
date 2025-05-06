import tfs
from aba_optimiser.config import TRACK_DATA_FILE, BPM_RANGE, RAMP_UP_TURNS
from aba_optimiser.utils import select_marker
from matplotlib import pyplot as plt

init_coords = tfs.read(TRACK_DATA_FILE, index="turn")
# Remove all rows that are not the BPM s.ds.r3.b1
start_bpm, end_bpm = BPM_RANGE.split("/")
start_coords = select_marker(init_coords, start_bpm)

# Remove all rows that have turn < num_ramp_up
start_coords = start_coords[start_coords.index > RAMP_UP_TURNS]

# Plot x, px phase space
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.scatter(start_coords["x"], start_coords["px"], s=1)
plt.xlabel("x")
plt.ylabel("px")
plt.title("x, px Phase Space")
plt.grid()
plt.subplot(2, 2, 2)
plt.scatter(
    start_coords["y"], 
    start_coords["py"], 
    s=1
)
plt.xlabel("y")
plt.ylabel("py")
plt.title("y, py Phase Space")
plt.grid()
plt.tight_layout()

end_coords = tfs.read(TRACK_DATA_FILE, index="turn")
# Remove all rows that are not the BPM s.ds.r3.b1
end_coords = select_marker(end_coords, "BPM.14R3.B1")
# Remove all rows that have turn < num_ramp_up
end_coords = end_coords[end_coords.index > RAMP_UP_TURNS]

# Plot x, px phase space
plt.subplot(2, 2, 3)
plt.scatter(end_coords["x"], end_coords["px"], s=1)
plt.xlabel("x")
plt.ylabel("px")
plt.title("x, px Phase Space")
plt.grid()
plt.subplot(2, 2, 4)
plt.scatter(
    end_coords["y"], 
    end_coords["py"], 
    s=1
)
plt.xlabel("y")
plt.ylabel("py")
plt.title("y, py Phase Space")
plt.grid()
plt.tight_layout()
plt.suptitle("Phase Space Comparison")
plt.subplots_adjust(top=0.88)  # Adjust the top margin to make room for the title
plt.savefig("phase_space_comparison.png", dpi=300)

plt.show()