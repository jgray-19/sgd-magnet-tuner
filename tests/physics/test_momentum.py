from pathlib import Path

import pytest
from pymadng import MAD


@pytest.fixture
def lhcb1_seq():
    """Fixture that reads the LHC B1 sequence file."""
    file_path = Path(__file__).parent.parent / "data" / "lhcb1.seq"
    with file_path.open() as f:
        return f.read()


@pytest.fixture(scope="module")
def twiss_data():
    """Module-level fixture that runs twiss on the LHC B1 sequence."""
    mad_script = f"""
"""
    script_path = Path(__file__).parent.parent / "twiss_script.mad"
    with script_path.open("w") as f:
        f.write(mad_script)

    mad = MAD()
    mad.send(mad_script)

    output_file = Path(__file__).parent.parent / "twiss_output.tfs"
    with output_file.open() as f:
        return f.read()
