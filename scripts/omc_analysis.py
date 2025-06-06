import numpy as np

import tfs
from lhcng.model_compressor import ModelCompressor
from omc3.hole_in_one import hole_in_one_entrypoint
from omc3.model_creator import create_instance_and_model

# from lhcng.model import create_model_dir
from aba_optimiser.config import (
    TRACK_DATA_FILE,
    POSITION_STD_DEV,
    module_path,
)

out_cols = ["name", "turn", "id", "eidx", "x", "px", "y", "py"]
numeric_cols = ["x", "px", "y", "py"]
data = tfs.read(TRACK_DATA_FILE)
data["x"] += np.random.normal(0, POSITION_STD_DEV, size=len(data))
data["y"] += np.random.normal(0, POSITION_STD_DEV, size=len(data))
DO_OMC_ANALYSIS = False
analysis_dir = module_path / 'analysis'
if DO_OMC_ANALYSIS:
    model_dir = module_path / 'models'
    model_dir.mkdir(exist_ok=True)
    if not (model_dir / 'twiss.tfs.bz2').exists():
        print("Creating LHC model for 2025...")
        modifier = "R2025aRP_A30cmC30cmA10mL200cm_Flat.madx"
        nat_tunes = [0.28, 0.31]
        accel = create_instance_and_model(
            accel="lhc",
            fetch="path",
            path=module_path / "mad_scripts" / 'acc-models-lhc',
            type='nominal',
            beam=1,
            year='2025',
            driven_excitation='acd',
            energy=6800,
            nat_tunes=nat_tunes,
            drv_tunes=nat_tunes,
            modifiers=[modifier],
            outputdir=model_dir,
        )
        compressor = ModelCompressor(model_dir)
        compressor.compress_model()

    out_cols = data.columns.tolist()
    cols_for_analysis = [col for col in out_cols if col not in ['px', 'py']]
    linfile_dir = TRACK_DATA_FILE.parent / "linfiles"
    linfile_dir.mkdir(exist_ok=True)

    tbt_file = TRACK_DATA_FILE.parent / 'noisy_omc3.tfs.bz2'
    tfs.write(tbt_file, data[cols_for_analysis])
    print('Running hole-in-one frequency analysis...')
    hole_in_one_entrypoint(
            harpy=True,
            files=[tbt_file],
            tbt_datatype='madng',
            outputdir=linfile_dir,
            to_write=["lin", "spectra"],
            opposite_direction=False,
            tunes=[0.28, 0.31, 0.0],
            natdeltas=[0.0, -0.0, 0.0],
            clean=False,
    )

    with ModelCompressor(model_dir): 
        print('Running hole-in-one optics analysis...')
        hole_in_one_entrypoint(
            files=[linfile_dir / tbt_file.name],
            outputdir=analysis_dir,
            optics=True,
            accel="lhc",
            beam=1,
            model_dir=model_dir,
            year='2025',
            compensation="none",
        )