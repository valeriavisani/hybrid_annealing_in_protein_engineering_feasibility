from pathlib import Path

# Root
BASE_DIR = Path(__file__).resolve().parents[1]

# Main directories
DATA_DIR = BASE_DIR / "data"
SCRIPTS_DIR = BASE_DIR / "scripts"

# PyRosetta
PYROSETTA_DIR = DATA_DIR / "pyRosetta"
ENERGY_FILES = PYROSETTA_DIR / "energy_files"
N_ROT_PER_RES = PYROSETTA_DIR / "n_rot_per_res"
PDB = PYROSETTA_DIR / "pdb"

# Plot_results
PLOT_RESULTS_DATA = DATA_DIR /"plot_results"
PLOT_RESULTS_SCRIPTS = SCRIPTS_DIR /"plot_results"
