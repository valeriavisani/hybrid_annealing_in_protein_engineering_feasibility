import pyrosetta
from pathlib import Path

from scripts.config import PDB

# Script to check that a PDB file has the correct number of residues and contains only backbone atoms

# To run: python -m data.pyRosetta.pdb.check_pdb

# Name of the PDB file: set PDB_FILE_NAME below (e.g.: PDB_FILE_NAME = "197residue.pdb")
PDB_FILE_NAME = "197residue.pdb"

pyrosetta.init(options="-mute all")

pdb_path = Path(PDB) / PDB_FILE_NAME

pose = pyrosetta.pose_from_pdb(str(pdb_path))

# Check the number of residues
print("Number of residues:", pose.total_residue())

# Check that there are no side-chain atoms
allowed_atoms = {"N", "CA", "C", "O", "OXT"}
unexpected = []
with open(pdb_path) as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_num = line[22:26].strip()
            if atom_name not in allowed_atoms:
                unexpected.append((res_num, res_name, atom_name))

if not unexpected:
    print("The PDB file contains only backbone atoms.")
else:
    print(f"The PDB file contains {len(unexpected)} non-backbone atoms!")
    for res_num, res_name, atom_name in unexpected:
        print(f"Residue {res_num} ({res_name}): {atom_name}")