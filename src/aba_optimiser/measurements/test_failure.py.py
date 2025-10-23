from __future__ import annotations

from pathlib import Path

from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.plotting.plot_optics_measurements import plot

analysis_dir_combined = Path("analysis_combined")
analysis_dir_separate = Path("analysis_separate")

files = [
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_52_18_410/Beam1@BunchTurn@2025_04_09@18_52_18_410.sdds",
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_48_27_464/Beam1@BunchTurn@2025_04_09@18_48_27_464.sdds",
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_51_14_983/Beam1@BunchTurn@2025_04_09@18_51_14_983.sdds",
    "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Measurements/Beam1@BunchTurn@2025_04_09@18_47_22_071/Beam1@BunchTurn@2025_04_09@18_47_22_071.sdds",
]
files = [Path(f) for f in files]

common_kwargs = {
    "files": files,
    "model_dir": "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Models/b1_flat_60_18cm",
}

harpy_kwargs = {
    "harpy": True,
    "unit": "mm",
    "driven_excitation": "acd",
    "first_bpm": "BPM.33L2.B1",
    "is_free_kick": False,
    "keep_exact_zeros": False,
    "max_peak": 0.02,
    "nattunes": [0.28, 0.31, 0.0],
    "num_svd_iterations": 3,
    "opposite_direction": False,
    "output_bits": 12,
    "peak_to_peak": 1e-08,
    "resonances": 4,
    "sing_val": 12,
    "svd_dominance_limit": 0.925,
    "tbt_datatype": "lhc",
    "to_write": ["lin", "spectra", "full_spectra", "bpm_summary"],
    "tolerance": 0.005,
    "tune_clean_limit": 1e-05,
    "tunes": [0.27, 0.322, 0.0],
    "turn_bits": 18,
    "turns": [0, 50000],
}

optics_kwargs = {
    "optics": True,
    "analyse_dpp": 0,
    "calibrationdir": "/afs/cern.ch/eng/sl/lintrack/LHC_commissioning2017/Calibration_factors_2017/Calibration_factors_2017_beam1",
    "chromatic_beating": False,
    "compensation": "equation",
    "coupling_method": 2,
    "coupling_pairing": 0,
    "isolation_forest": False,
    # "nonlinear": ["rdt"],
    "only_coupling": False,
    "range_of_bpms": 11,
    # "rdt_magnet_order": 4,
    "second_order_dispersion": False,
    "three_bpm_method": False,
    "three_d_excitation": False,
    "union": False,
    "accel": "lhc",
    "ats": False,
    "beam": 1,
    "dpp": 0.0,
    "xing": False,
    "year": "2025",
}

hole_in_one_entrypoint(
    outputdir=analysis_dir_separate / "lin_files", **common_kwargs, **harpy_kwargs
)

hole_in_one_entrypoint(
    outputdir=analysis_dir_separate, **common_kwargs, **optics_kwargs
)

hole_in_one_entrypoint(
    outputdir=analysis_dir_combined, **common_kwargs, **harpy_kwargs, **optics_kwargs
)

figs = plot(
    folders=[
        analysis_dir_combined,
        analysis_dir_separate,
        "/user/slops/data/LHC_DATA/OP_DATA/Betabeat/2025-04-09/LHCB1/Results/b1_flat_60_18cm_after_coupling_cor",
    ],
    labels=[
        "Combined Analysis",
        "Separate Analysis",
        "Measurement online",
    ],
    combine_by=["files"],  # to compare folder1 and folder2
    # output='output_directory',
    delta=True,  # delta from reference
    optics_parameters=[
        # "orbit",
        # "beta_phase",
        # "beta_amplitude",
        # "phase",
        "total_phase",
        # "f1001_x",
        # "f1010_x",
    ],
    x_axis="location",  # or 'phase-advance'
    ip_positions="LHCB1",
    suppress_column_legend=True,
    show=True,
    ncol_legend=2,
)
