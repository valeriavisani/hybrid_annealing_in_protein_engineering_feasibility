# hybrid_annealing_in_protein_engineering_feasibility

Code and data accompanying the paper

> **Hybrid Quantum-Classical Annealing for Protein Side-Chain Optimization: A Feasibility Study**
> Valeria B. C. Visani, Simone Reale, Alfonso Gautieri, Elisabetta Di Nitto.
> ArXiv preprint: [arXiv:2611.08696](https://arxiv.org/abs/2611.08696), 2026.
>
> Submitted to the IEEE International Symposium on Circuits and Systems (ISCAS) 2027,
> Special Session *"Bridging Quantum Circuits and Real-World Applications: Algorithms,
> Architectures, and Hardware Constraints"* (under review).

The repository benchmarks the D-Wave Leap hybrid CQM solver (5 s per run) against Gurobi
(10 s, 1000 s and a 4 h reference run) on 40 Protein Side-Chain Optimization (PSCO) instances.
The instances are built from 20 PDB structures (10 to 197 residues), each with 5 and 10 rotamers
per residue (50 to 1970 binary variables).

## Problem

Protein Side-Chain Optimization (PSCO) is the problem of choosing, for every residue of a protein,
one conformation of its side chain (a *rotamer*) from a discrete library, while the backbone is kept
fixed. The goal is to find the combination with the lowest total energy. Here the energy is computed by Rosetta
with the `ref2015` score function and is assumed to be pairwise additive: it is the sum of the
energy of each residue with a specific rotamer in the fixed backbone (one-body term) and of the interaction energy of each
pair of residues with specific rotamers (two-body term) in the fixed backbone.

With a binary variable `x_{i,r}` equal to 1 if residue `i` takes rotamer `r` and 0 otherwise, PSCO
is a binary quadratic program with one-hot constraints:

```
min   sum_i sum_r E_ii(r) x_{i,r}  +  sum_{i<j} sum_{r,s} E_ij(r,s) x_{i,r} x_{j,s}
s.t.  sum_r x_{i,r} = 1   for every residue i
      x_{i,r} = 0         for r > n_i
```

- `E_ii(r)`: one-body energy of rotamer `r` of residue `i`.
- `E_ij(r,s)`: two-body energy between rotamer `r` of residue `i` and rotamer `s` of residue `j`.
  It is zero for pairs of residues that do not interact.
- `n_i`: real number of rotamers available for residue `i`. Each residue is given the same number of
  variables (5 or 10 in our instances), so residues with fewer rotamers are zero-padded and their
  unused variables are fixed to zero.

An instance with `N` residues and `R` rotamers per residue has `N·R` binary variables. The energy
terms `E_ii` and `E_ij` are stored in `data/pyRosetta/energy_files`, and the values `n_i` in
`data/pyRosetta/n_rot_per_res`.

## Repository structure

```
.
├── data/
│   ├── pyRosetta/
│   │   ├── pdb/                  input PDB files (sources: pdb_sources.txt) and check_pdb.py
│   │   ├── energy_files/         one-body and two-body energy terms (CSV)
│   │   └── n_rot_per_res/        real number of rotamers of each residue (CSV)
│   ├── reference_results_{5,10}rot.csv            Gurobi reference (default config., 4 h)
│   ├── gurobi_results_{5,10}rot_{10s,1000s}.csv   Gurobi, custom configuration (10 s / 1000 s)
│   ├── hybrid_annealing_results_{5,10}rot.csv     D-Wave hybrid CQM solver (5 s)
│   ├── reference.csv             output of new runs of gurobi_reference.py
│   ├── gurobi.csv                output of new runs of gurobi.py
│   ├── hybrid_annealing.csv      output of new runs of hybrid_annealing.py
│   ├── hybrid_annealing_samplesets/   full sampleset returned by each hybrid annealing run
│   └── plot_results/             figure and tables produced by the plotting scripts
├── scripts/
│   ├── config.py                 paths and shared settings
│   ├── pyRosetta/
│   │   ├── create_energy_files_official.py   energy files, instances with up to 157 residues
│   │   └── create_energy_files_manual.py     energy files, instances with more than 157 residues
│   ├── gurobi_reference.py       Gurobi reference run (default configuration, 4 h)
│   ├── gurobi.py                 Gurobi runs (custom configuration, 10 s / 1000 s)
│   ├── hybrid_annealing.py       D-Wave Leap hybrid CQM runs
│   └── plot_results/
│       ├── quality_plot_feasibility.py   gap-from-reference figure
│       └── qpu_time.py                   QPU access time statistics
├── third_party/
│   └── LICENSE_quantum-protein-folding   Apache-2.0 license of the stfc repository
├── LICENSE
├── README.md
└── requirements.txt

```

## Requirements

- Python 3.12
- [PyRosetta](https://www.pyrosetta.org/) (installed with `pyrosetta-installer`, see below)
- `gurobipy` (license needed)
- `dwave-ocean-sdk` and a [D-Wave Leap](https://cloud.dwavesys.com/leap/) account with an API token

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyRosetta is not pinned in requirements.txt, because its wheels depend on the platform and the Python version. `pyrosetta-installer` is installed with the requirements; use it to install the PyRosetta build that matches your system:

```bash
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

## Usage

All scripts are run from the repository root as modules (`python -m ...`).

### 1. Generate the energy files

```bash
python -m scripts.pyRosetta.create_energy_files_official -p <pdb_filename> -r <num_rotamers> -i <starting_rotamer_index>
python -m scripts.pyRosetta.create_energy_files_manual   -p <pdb_filename> -r <num_rotamers> -i <starting_rotamer_index>
```

- `-p` PDB file name in `data/pyRosetta/pdb`, `-r` number of rotamers per residue, `-i` index of the first rotamer of the window.
- The *official* script reads the energies from Rosetta's `InteractionGraphSummaryMetric`. It was used for the instances with up to 157 residues.
- The *manual* script extracts the same terms by traversing the Rosetta interaction graph. It was used for the instances with more than 157 residues.
- Residues with fewer real rotamers than requested are zero-padded; the real counts are saved in `data/pyRosetta/n_rot_per_res`.
- The structure relaxation (FastRelax) is not seeded, so rerunning the scripts can give slightly different energies. For reproducibility use the energy files provided in this repository.

### 2. Solve the instances

```bash
python -m scripts.gurobi_reference -res <num_res> -rot <num_rot>
python -m scripts.gurobi -res <num_res> -rot <num_rot> [-s <seed>]
python -m scripts.hybrid_annealing -res <num_res> -rot <num_rot> [-run_id <run_id>]
```

The hybrid solver needs the API token of your Leap account. Create the file `scripts/configto.py`
(it is not included in the repository, it is listed in `.gitignore`) containing:

```python
DWAVE_TOKEN = "<your Leap API token>"
```

Gurobi custom configuration: `MIPFocus=1`, `Heuristics=0.5`, `MIPGap=0.1`.

Where the outputs go:

- The results of a new run are written to `data/reference.csv` (`gurobi_reference.py`), `data/gurobi.csv` (`gurobi.py`) and `data/hybrid_annealing.csv` (`hybrid_annealing.py`). These three files are the working files for new executions.
- The files `reference_results_*`, `gurobi_results_*` and `hybrid_annealing_results_*` contain the results reported in the paper and are never modified by the scripts. The plotting scripts read only these files, so new runs do not change the figure or the statistics of the paper.
- Each hybrid annealing run also saves the full sampleset returned by the solver in `data/hybrid_annealing_samplesets/`.

### 3. Reproduce the figure and the statistics

```bash
python -m scripts.plot_results.quality_plot_feasibility [--shared_y]
python -m scripts.plot_results.qpu_time [--min_res MIN_RES] [--max_res MAX_RES]
```

- `quality_plot_feasibility` writes `data/plot_results/quality_plot_feasibility.png`: percentage gap from the reference energy versus instance size (mean, min and max over the runs).
- `qpu_time` prints the mean QPU access time per run and its share of the total run time, and writes `data/plot_results/qpu_time_per_instance.csv`.

## Data

Results of the paper (fixed files):

| File | Content |
|---|---|
| `reference_results_{rot}rot.csv` | Reference energy (Gurobi, default configuration, 4 h) and solver status per instance |
| `gurobi_results_{rot}rot_{10s,1000s}.csv` | Gurobi results, custom configuration, one row per run |
| `hybrid_annealing_results_{rot}rot.csv` | D-Wave hybrid CQM results (energy, run time, QPU access time), one row per run |

Output of new executions (working files):

| File | Content |
|---|---|
| `reference.csv`, `gurobi.csv`, `hybrid_annealing.csv` | Results of the runs launched with `gurobi_reference.py`, `gurobi.py` and `hybrid_annealing.py`, in the same format as the corresponding fixed files |
| `hybrid_annealing_samplesets/ha_{N}res_{R}rot_run{id}.csv` | Full sampleset of one hybrid annealing run (`N` residues, `R` rotamers, run `id`): one row per returned sample, with columns `num_res`, `num_rot`, `gs_bitstring` (binary assignment of the `N·R` variables, written as `N` groups of `R` bits separated by spaces, one group per residue in residue order; e.g. `01 01 01 10` for 4 residues and 2 rotamers) and `gs_energy [REU]` |

Input data:

| File | Content |
|---|---|
| `data/pyRosetta/energy_files/` | `{rot}rot_{N}res_one_body_terms.csv` and `..._two_body_terms.csv` (`*_manual.csv` for the instances generated with the manual script) |
| `data/pyRosetta/n_rot_per_res/` | `nrot_per_res_{N}res.csv`: real number of rotamers of each residue |

The source entries of all input structures are listed in [`pdb_sources.txt`](data/pyRosetta/pdb/pdb_sources.txt).

Energies are in Rosetta Energy Units (REU), score function used: `ref2015`.

## Machine used

Energy files and Gurobi runs: Intel Xeon Silver 4114 (16 vCPUs, 2.2 GHz, 62 GiB RAM).

## Third-party material

Part of this repository is derived from
[stfc/quantum-protein-folding](https://github.com/stfc/quantum-protein-folding), released under the
Apache License 2.0 and associated with the paper:

> *Quantum Algorithm for Protein Side-Chain Optimisation: Comparing Quantum to Classical Methods*,
> A. Agathangelou, D. Manawadu, I. Tavernelli,
> [arXiv:2507.19383](https://arxiv.org/abs/2507.19383) (2025).

- `scripts/pyRosetta/create_energy_files_official.py` and `scripts/pyRosetta/create_energy_files_manual.py` are adapted from their energy-file generation script. The header of each file lists the modifications.
- `data/pyRosetta/pdb/4residue.pdb` was taken from their repository without modification. It was used only for testing and does not appear in the results of the paper.
- As required by the Apache License 2.0, a copy of their license is included in [`third_party/LICENSE_quantum-protein-folding`](third_party/LICENSE_quantum-protein-folding). That file applies only to the material derived from their repository, not to the rest of this repository.


## Citation

If you use this repository, please cite:

```bibtex
@misc{visani2026,
      title={Hybrid Quantum-Classical Annealing for Protein Side-Chain Optimization: A Feasibility Study}, 
      author={Valeria B. C. Visani and Simone Reale and Alfonso Gautieri and Elisabetta Di Nitto},
      year={2026},
      eprint={2611.08696},
      archivePrefix={arXiv},
      primaryClass={cs.ET},
      url={https://arxiv.org/abs/2611.08696}, 
}
```