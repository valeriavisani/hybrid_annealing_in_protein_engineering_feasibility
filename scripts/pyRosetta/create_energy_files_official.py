import io
import re
import argparse
import contextlib
import pandas as pd
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta import init, pose_from_pdb, get_score_function
from pyrosetta.rosetta.protocols.quantum_annealing import InteractionGraphSummaryMetric

## To run: python -m scripts.pyRosetta.create_energy_files_official -p <pdb_filename> -r <num_rotamers> -i <starting_rotamer_index>
## Note: <pdb_filename> must contain the extension

from ..config import PDB, ENERGY_FILES, N_ROT_PER_RES

# Given a PDB file, the number of rotamers to consider and the index of the first rotamer,
# this script extracts the one-body and two-body energy terms.
# The energy terms are extracted with the official Rosetta module InteractionGraphSummaryMetric:
# protocols::quantum_annealing::InteractionGraphSummaryMetric
# (https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/SimpleMetrics/simple_metric_pages/InteractionGraphSummaryMetric)
#
# USE IN THE PAPER
# Used for the instances with up to 157 residues. Larger instances ran out of memory with this
# script on the machine used (Intel Xeon Silver 4114, 16 cores, 2.2 GHz, 62 GiB RAM) and were
# generated with create_energy_files_manual.py, which extracts the same terms directly from the
# interaction graph.
#
# ORIGIN AND LICENSE
# Parts of this script (prepare_pose, the layout of the output files and the zero-padding convention)
# are adapted from the energy-file generation script of the repository
# https://github.com/stfc/quantum-protein-folding (code authors: D. Manawadu, A. Agathangelou;
# Apache License 2.0, copy in third_party/LICENSE-quantum-protein-folding),
# associated with the paper:
#   A. Agathangelou, D. Manawadu, I. Tavernelli, "Quantum Algorithm for Protein Side-Chain
#   Optimisation: Comparing Quantum to Classical Methods", arXiv:2507.19383 (2025).
#
# CHANGES WITH RESPECT TO THE ORIGINAL
# 1. The energy terms are parsed from the output of InteractionGraphSummaryMetric instead of being
#    read by traversing the interaction graph manually.
# 2. The model of the original paper keeps only the two-body terms between rotamers of adjacent
#    residues (nearest neighbours along the sequence). Here the two-body terms are extracted for
#    every pair of residues, including long-range ones: all the two-body terms of the rotamer
#    window are written, and the pairs of residues without an edge in the interaction graph (i.e.
#    that do not interact) get energy zero, so that the output files have the same structure for
#    every instance.
# 3. The number of real rotamers of each residue is saved, and the energy terms of the rotamers that
#    do not exist are set to zero (padding). The corresponding variables are fixed to zero in the
#    model (x_{i,r} = 0 for r > n_i).
# 4. Minor changes: PyRosetta output is silenced, output paths and file names.

def parse_args():
    # function to read the user input

    parser = argparse.ArgumentParser(
        description="Create energy files for protein folding using the official "
        "Rosetta InteractionGraphSummaryMetric module."
    )
    parser.add_argument(
        "-p", "--pdb", required=True, help="PDB filename (with .pdb extension)"
    )
    parser.add_argument(
        "-r",
        "--num_rot",
        type=int,
        required=True,
        help="Number of rotamers to record per residue (padding size for the CQM; "
        "residues with fewer real rotamers are zero-padded, as in the original script)",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help="Starting index for rotamer selection (native Rosetta index, 1-based)",
    )

    return parser.parse_args()


def prepare_pose(pdb_path):
    # function that, given the path of the PDB file, returns a pose whose geometry
    # is consistent with the force field (FastRelax)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        pose = pose_from_pdb(str(pdb_path))
        sfxn = get_score_function(True)
        relax = FastRelax()
        relax.set_scorefxn(sfxn)
        relax.apply(pose)
    return pose, sfxn


def get_ig_summary_text(pose, sfxn):
    # function that, given a pose and a scoring function, builds the interaction graph
    # with the official PyRosetta module and returns its textual summary

    # create a default TaskFactory
    task_factory = TaskFactory()

    # create the InteractionGraphSummaryMetric object
    metric = InteractionGraphSummaryMetric()
    # set the scoring function
    metric.set_scorefunction(sfxn)
    # set the TaskFactory
    metric.set_task_factory(task_factory)
    # short_version = True  ->  only the summary of the interaction graph (one-body/two-body),
    # without the information needed to rebuild the pose
    metric.set_short_version(True)

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        summary_text = metric.calculate(pose)

    # print(f"Raw string length: {len(summary_text):,} characters")
    # print(f"Number of lines: {summary_text.count(chr(10)):,}")
    # print(summary_text) # > output.txt
    return summary_text


def parse_ig_summary(summary_text):
    # function that, given the text returned by InteractionGraphSummaryMetric,
    # extracts the one-body and two-body terms and the real number of rotamers per residue.

    one_body = {}   # (res, rot) -> energy
    two_body = {}   # (res_i, rot_i, res_j, rot_j) -> energy
    real_rotamers = {}  # res -> set of indices of the rotamers that actually exist

    for raw_line in summary_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = re.split(r"\s+", line)

        try:
            if len(tokens) == 3:
                res = int(tokens[0])
                rot = int(tokens[1])
                energy = float(tokens[2])
                one_body[(res, rot)] = energy
                real_rotamers.setdefault(res, set()).add(rot)

            elif len(tokens) == 5:
                res_i = int(tokens[0])
                rot_i = int(tokens[1])
                res_j = int(tokens[2])
                rot_j = int(tokens[3])
                energy = float(tokens[4])
                two_body[(res_i, rot_i, res_j, rot_j)] = energy
                two_body[(res_j, rot_j, res_i, rot_i)] = energy  # symmetric

            else:
                # line that matches neither of the two data sections: ignored
                continue

        except ValueError:
            # non-numeric line (comment/header without '#', etc.): ignored
            continue

    return one_body, two_body, real_rotamers


def save_num_rotamers_per_residue(real_rotamers, num_res):
    # function that, given the dictionary of real rotamers, returns a dataframe
    # that associates each residue with its real number of rotamers

    data = []
    for res_i in range(1, num_res + 1):
        nrot = len(real_rotamers.get(res_i, set()))
        data.append({"residue": res_i, "num_rotamers": nrot})

    return pd.DataFrame(data)


def extract_one_body_data(one_body, num_rot, index, num_res):
    # function that, given one_body, num_rot, index, num_res,
    # returns a dataframe that associates each possible residue-rotamer choice with its one-body energy term

    data_list = []
    for res_i in range(1, num_res + 1):
        for rot_i in range(index, index + num_rot):
            energy = one_body.get((res_i, rot_i), 0.0)
            data_list.append({"res i": res_i, "rot A_i": rot_i, "E_ii": energy})

    return pd.DataFrame(data_list)


def extract_two_body_data(two_body, num_rot, index, num_res):
    # function that, given two_body, num_rot, index, num_res,
    # returns a dataframe that associates each possible pair of residue-rotamer choices with its two-body energy term

    data_list = []
    for res_i in range(1, num_res):
        for res_j in range(res_i + 1, num_res + 1):
            for rot_i in range(index, index + num_rot):
                for rot_j in range(index, index + num_rot):
                    energy = two_body.get((res_i, rot_i, res_j, rot_j), 0.0)
                    data_list.append(
                        {
                            "res i": res_i,
                            "res j": res_j,
                            "rot A_i": rot_i,
                            "rot B_j": rot_j,
                            "E_ij": energy,
                        }
                    )

    return pd.DataFrame(data_list)


def main():

    # store the user input in variables
    args = parse_args()
    pdb_path = PDB / f"{args.pdb}"
    num_rot = args.num_rot
    index = args.index

    if not pdb_path.exists():
        raise FileNotFoundError(f"{pdb_path} does not exist.")

    # start PyRosetta
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        init(options="-mute all")

    # create the pose from the input PDB and the default scoring function
    pose, sfxn = prepare_pose(pdb_path)

    # store the number of residues in a variable
    num_res = pose.total_residue()

    # build the summary of the metrics extracted by InteractionGraphSummaryMetric
    summary_text = get_ig_summary_text(pose, sfxn)

    # extract the one-body terms, the two-body terms and the real number of rotamers of each residue
    one_body, two_body, real_rotamers = parse_ig_summary(summary_text)

    # file: real number of rotamers per residue
    df_n_rot_per_res = save_num_rotamers_per_residue(real_rotamers, num_res)
    out_path_n_rot = N_ROT_PER_RES / f"nrot_per_res_{num_res}res.csv"
    df_n_rot_per_res.to_csv(out_path_n_rot, index=False)
    print(f"\n\nSaved rotamer count per residue to: {out_path_n_rot}")

    # files: one-body and two-body terms
    df_one_body = extract_one_body_data(one_body, num_rot, index, num_res)
    df_two_body = extract_two_body_data(two_body, num_rot, index, num_res)

    out_path_ob = ENERGY_FILES / f"{num_rot}rot_{num_res}res_one_body_terms.csv"
    df_one_body.to_csv(out_path_ob, index=False)

    out_path_tb = ENERGY_FILES / f"{num_rot}rot_{num_res}res_two_body_terms.csv"
    df_two_body.to_csv(out_path_tb, index=False)

    print(f"\n\nEnergy data saved to: {out_path_ob} and {out_path_tb}")


if __name__ == "__main__":
    main()