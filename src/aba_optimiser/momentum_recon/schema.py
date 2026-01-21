
from dataclasses import dataclass

# Core column names used across the codebase
CORE_ID_COLS = ("name", "turn")
CORE_POS_COLS = ("x", "y")
CORE_MOM_COLS = ("px", "py")
CORE_VAR_COLS = ("var_x", "var_y")

# Neighbor suffixes and planes
SUFFIX_PREV = "p"
SUFFIX_NEXT = "n"
PLANE_X = "x"
PLANE_Y = "y"


@dataclass(frozen=True)
class NeighborNames:
    bpm_x: str
    bpm_y: str
    delta_x: str
    delta_y: str
    delta_x_err: str
    delta_y_err: str
    x: str
    y: str
    var_x: str
    var_y: str
    dx: str
    dy: str


PREV = NeighborNames(
    bpm_x="bpm_x_p",
    bpm_y="bpm_y_p",
    delta_x="delta_x_p",
    delta_y="delta_y_p",
    delta_x_err="delta_x_p_err",
    delta_y_err="delta_y_p_err",
    x="x_p",
    y="y_p",
    var_x="var_x_p",
    var_y="var_y_p",
    dx="dx_p",
    dy="dy_p",
)

NEXT = NeighborNames(
    bpm_x="bpm_x_n",
    bpm_y="bpm_y_n",
    delta_x="delta_x_n",
    delta_y="delta_y_n",
    delta_x_err="delta_x_n_err",
    delta_y_err="delta_y_n_err",
    x="x_n",
    y="y_n",
    var_x="var_x_n",
    var_y="var_y_n",
    dx="dx_n",
    dy="dy_n",
)

POSITION_COLS = ("x", "y")
VARIANCE_COLS = ("var_x", "var_y")
