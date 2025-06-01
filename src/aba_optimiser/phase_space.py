import numpy as np
import tfs
from aba_optimiser.mad_interface import MadInterface
from aba_optimiser.config import SEQUENCE_FILE, SEQ_NAME, BPM_RANGE

class PhaseSpaceDiagnostics:
    def __init__(self, bpm, x_data, px_data, y_data, py_data, num_points=500):
        self.bpm = bpm
        self.num_points = num_points
        self.x_data = x_data
        self.px_data = px_data
        self.y_data = y_data
        self.py_data = py_data

        # Get Twiss parameters
        self._load_twiss()

        # Compute emittances
        self._compute_emittances()

        # Precompute gamma
        self.gammax = (1 + self.alfax**2) / self.betax
        self.gammay = (1 + self.alfay**2) / self.betay

    def _load_twiss(self):
        mad = MadInterface(SEQUENCE_FILE, BPM_RANGE, stdout="/dev/null", redirect_sterr=True)
        mad.mad.send(f"tws = twiss{{sequence=MADX.{SEQ_NAME}, observe=1}}")
        tws: tfs.TfsDataFrame = mad.mad.tws.to_df().set_index("name")

        self.betax = tws.loc[self.bpm, "beta11"]
        self.alfax = tws.loc[self.bpm, "alfa11"]
        self.betay = tws.loc[self.bpm, "beta22"]
        self.alfay = tws.loc[self.bpm, "alfa22"]

    def _compute_emittances(self):
        cov_x = np.cov(self.x_data, self.px_data)
        self.emit_x = np.sqrt(cov_x[0, 0] * cov_x[1, 1] - cov_x[0, 1]**2)

        cov_y = np.cov(self.y_data, self.py_data)
        self.emit_y = np.sqrt(cov_y[0, 0] * cov_y[1, 1] - cov_y[0, 1]**2)

    def compute_residuals(self):
        inv_x = self.gammax * self.x_data**2 + 2 * self.alfax * self.x_data * self.px_data + self.betax * self.px_data**2
        inv_y = self.gammay * self.y_data**2 + 2 * self.alfay * self.y_data * self.py_data + self.betay * self.py_data**2

        residuals_x = (inv_x - self.emit_x) / self.emit_x
        residuals_y = (inv_y - self.emit_y) / self.emit_y

        std_x = np.std(residuals_x)
        std_y = np.std(residuals_y)

        return std_x, std_y

    def ellipse_points(self):
        theta = np.linspace(0, 2*np.pi, self.num_points)

        x = np.sqrt(2 * self.emit_x * self.betax) * np.cos(theta)
        px = -np.sqrt(2 * self.emit_x / self.betax) * (self.alfax * np.cos(theta) + np.sin(theta))

        y = np.sqrt(2 * self.emit_y * self.betay) * np.cos(theta)
        py = -np.sqrt(2 * self.emit_y / self.betay) * (self.alfay * np.cos(theta) + np.sin(theta))

        return x, px, y, py

    def ellipse_sigma(self, sigma_level=1.0):
        std_x, std_y = self.compute_residuals()

        emit_x_scaled = self.emit_x * (1 + sigma_level * std_x)
        emit_y_scaled = self.emit_y * (1 + sigma_level * std_y)

        theta = np.linspace(0, 2*np.pi, self.num_points)

        x_upper = np.sqrt(2 * emit_x_scaled * self.betax) * np.cos(theta)
        px_upper = -np.sqrt(2 * emit_x_scaled / self.betax) * (self.alfax * np.cos(theta) + np.sin(theta))

        y_upper = np.sqrt(2 * emit_y_scaled * self.betay) * np.cos(theta)
        py_upper = -np.sqrt(2 * emit_y_scaled / self.betay) * (self.alfay * np.cos(theta) + np.sin(theta))

        return x_upper, px_upper, y_upper, py_upper