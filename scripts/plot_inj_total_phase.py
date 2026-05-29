import shutil
from pathlib import Path

from omc3.plotting.plot_optics_measurements import plot
from omc3.plotting.utils.windows import PlotWidget, SimpleTabWindow

ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = [
    "fake_measurements_inj_tunes_matched.zip",
    "fake_measurements_inj_no_tune_match.zip",
    "fake_measurements_inj_no_tune_match_model_onmom.zip",
    "fake_measurements_inj_no_tune_match_model_onmom_drop_ks.zip",
    "fake_measurements_inj_tunes_matched_drop_ks_model_onmom.zip",
    "fake_measurements_inj_tunes_matched_drop_ks.zip",
]
ARCHIVES = [ROOT / archive for archive in ARCHIVES]

cleanup_dirs: list[Path] = []
window = SimpleTabWindow("Inj Total Phase", size=(1800, 1100))
try:
    for archive in ARCHIVES:
        extracted = ROOT / "fake_measurements_inj"
        target = ROOT / archive.stem
        shutil.rmtree(extracted, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        shutil.unpack_archive(archive, ROOT)
        extracted.rename(target)
        cleanup_dirs.append(target)
        freq_folders = sorted(
            folder for folder in target.iterdir() if (folder / "total_phase_x.tfs").exists()
        )
        figs = plot(
            folders=freq_folders,
            labels=[folder.name for folder in freq_folders],
            delta=True,
            x_axis="location",
            optics_parameters=["total_phase"],
            ip_positions="LHCB1",
            suppress_column_legend=True,
            combine_by=["files"],
            show=False,
        )
        window.add_tab(PlotWidget(next(iter(figs.values())), title=archive.stem))
    window.show()
finally:
    for folder in cleanup_dirs:
        shutil.rmtree(folder, ignore_errors=True)
