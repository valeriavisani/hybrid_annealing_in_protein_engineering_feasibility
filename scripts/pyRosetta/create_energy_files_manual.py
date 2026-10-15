import io
import argparse
import pyrosetta
import contextlib
import pandas as pd
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta import init, pose_from_pdb, get_score_function
from pyrosetta.rosetta.core.pack.rotamer_set import RotamerSets
from pyrosetta.rosetta.core.pack.interaction_graph import InteractionGraphFactory

## To run: python -m scripts.pyRosetta.create_energy_files_manual -p <pdb_filename> -r <num_rotamers> -i <starting_rotamer_index>
## Note: <pdb_filename> must contain the extension

from ..config import PDB, ENERGY_FILES, N_ROT_PER_RES

# Given a PDB file, the number of rotamers to consider and the index of the first rotamer,
# this script builds the Rosetta InteractionGraph and extracts the one-body and two-body
# energy terms by traversing the InteractionGraph manually.
#
# Rosetta also exposes the module InteractionGraphSummaryMetric, which extracts metrics from the
# InteractionGraph, including its one-body and two-body energy terms. The script
# create_energy_files_official.py produces the energy files with that module.
#
# USE IN THE PAPER
# Used for the instances with more than 157 residues, for which create_energy_files_official.py
# ran out of memory on the machine used (Intel Xeon Silver 4114, 16 cores, 2.2 GHz, 62 GiB RAM).
# The instances with up to 157 residues were generated with create_energy_files_official.py.
# This script extracts the same terms, reading the selected window of rotamers directly from the
# InteractionGraph.
#
# ORIGIN AND LICENSE
# Derived from the energy-file generation script of the repository
# https://github.com/stfc/quantum-protein-folding (code authors: D. Manawadu, A. Agathangelou;
# Apache License 2.0, copy in third_party/LICENSE-quantum-protein-folding),
# associated with the paper:
#   A. Agathangelou, D. Manawadu, I. Tavernelli, "Quantum Algorithm for Protein Side-Chain
#   Optimisation: Comparing Quantum to Classical Methods", arXiv:2507.19383 (2025).
#
# CHANGES WITH RESPECT TO THE ORIGINAL
# 1. The model of the original paper keeps only the two-body terms between rotamers of adjacent
#    residues (nearest neighbours along the sequence). Here the two-body terms are extracted for
#    every pair of residues, including long-range ones: all the two-body terms of the rotamer
#    window are written, and the pairs of residues without an edge in the InteractionGraph (i.e.
#    that do not interact) get energy zero, so that the output files have the same structure for
#    every instance.
# 2. The real number of rotamers of each residue is saved; the terms of the rotamers beyond this
#    number are set to zero, and the corresponding variables are fixed to zero in the model
#    (constraint x_{i,r} = 0 for r > n_i).
# 3. Minor changes: PyRosetta output is silenced, output paths and file names.

def parse_args():
    # function to read the user input

    parser = argparse.ArgumentParser(
        description="Create energy files for protein folding."
    )
    parser.add_argument(
        "-p", "--pdb", required=True, help="PDB filename (with .pdb extension)"
    )
    parser.add_argument(
        "-r",
        "--num_rot",
        type=int,
        required=True,
        help="Number of rotamers to record per residue",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help="Starting index for rotamer selection",
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


def build_rotamers(pose, sfxn):
    # function that, given a pose and a scoring function, returns the rotamer sets and the interaction graph

    # create a PackerTask from the pose; for each residue this object holds information
    # about which operations are allowed on that residue (a list of TaskOperations per residue)
    task = TaskFactory.create_packer_task(pose)

    # update the neighbor information of each residue of the pose
    pose.update_residue_neighbors()

    # set up the scoring function to compute energies for this pose and this PackerTask
    sfxn.setup_for_packing(pose, task.designing_residues(), task.designing_residues())

    # create a graph from the pose, the scoring function and the task
    graph = pyrosetta.rosetta.core.pack.create_packer_graph(pose, sfxn, task)

    # create the RotamerSets object
    rotsets = RotamerSets()

    # add to rotsets the information contained in the task
    rotsets.set_task(task)

    # build the list of possible rotamers of each residue; the information is stored in rotsets.
    # The rotamer lists are built from: the pose, the scoring function and the graph.
    rotsets.build_rotamers(pose, sfxn, graph)

    # prepare the rotamers for packing
    rotsets.prepare_sets_for_packing(pose, sfxn)

    # create the interaction graph between the rotamers
    ig = InteractionGraphFactory.create_interaction_graph(
        task, rotsets, pose, sfxn, graph
    )

    # compute the interaction energies between rotamers and store them in the interaction graph
    rotsets.compute_energies(pose, sfxn, graph, ig, 1)

    return rotsets, ig


def save_num_rotamers_per_residue(rotsets, num_res):
    # function that, given rotsets and num_res, returns a dataframe that associates each residue with its real number of rotamers

    data = []

    # loop over all residues
    for res_i in range(1, num_res + 1):

        # store in a variable the rotamer set of the i-th residue
        rot_set = rotsets.rotamer_set_for_residue(res_i)
        # store in a variable the number of rotamers in the rotamer set of the i-th residue
        nrot = rot_set.num_rotamers() if rot_set else 0

        data.append(
            {
                "residue": res_i,
                "num_rotamers": nrot,
            }
        )

    df = pd.DataFrame(data)

    return df


def extract_one_body_data(rotsets, ig, num_rot, index, num_res):
    # function that, given rotsets, the interaction graph, num_rot, the rotamer index and num_res,
    # returns a dataframe that associates each possible residue-rotamer choice with its one-body energy term

    data_list = []

    # loop over all residues
    for res_i in range(1, num_res + 1):

        # store in a variable the index that identifies the i-th residue in the interaction graph
        molten_i = rotsets.resid_2_moltenres(res_i)
        # store in a variable the rotamer set of the i-th residue
        rot_set_i = rotsets.rotamer_set_for_residue(res_i)
        # store in a variable the number of rotamers in the rotamer set of the i-th residue
        nrot_i = rot_set_i.num_rotamers() if rot_set_i else 0


        # loop over the rotamers of the i-th residue, from the given starting index for num_rot rotamers
        for rot_i in range(index, index + num_rot):

            # if the current rotamer index is greater than the number of rotamers
            # of the residue, set the corresponding energy term to zero
            if (
                rot_i >= nrot_i
            ):
                energy = 0.0

            # otherwise store the energy term associated with the node of the interaction graph
            # that corresponds to this residue-rotamer choice
            else:
                energy = ig.get_one_body_energy_for_node_state(molten_i, rot_i)

            data_list.append({
                "res i": res_i,
                "rot A_i": rot_i,
                "E_ii": energy
                })

    df = pd.DataFrame(data_list)
    return df


def extract_two_body_data(rotsets, ig, num_rot, index, num_res):
    # function that, given rotsets, the interaction graph, num_rot, the rotamer index and num_res,
    # returns a dataframe that associates each possible pair of residue-rotamer choices with its two-body energy term

    data_list = []

    # loop over all residues
    for res_i in range(1, num_res):

        # store in a variable the index that identifies the i-th residue in the interaction graph
        molten_i = rotsets.resid_2_moltenres(res_i)
        # store in a variable the rotamer set of the i-th residue
        rot_set_i = rotsets.rotamer_set_for_residue(res_i)
        # store in a variable the number of rotamers in the rotamer set of the i-th residue
        nrot_i = rot_set_i.num_rotamers() if rot_set_i else 0

        # loop over all the residues that follow the i-th residue
        for res_j in range(res_i + 1, num_res + 1):

            # store in a variable the index that identifies the j-th residue in the interaction graph
            molten_j = rotsets.resid_2_moltenres(res_j)
            # store in a variable the rotamer set of the j-th residue
            rot_set_j = rotsets.rotamer_set_for_residue(res_j)
            # store in a variable the number of rotamers in the rotamer set of the j-th residue
            nrot_j = rot_set_j.num_rotamers() if rot_set_j else 0

            # store in a boolean variable whether an edge exists between the nodes of the two residues
            edge_exists = ig.find_edge(molten_i, molten_j)

            # loop over the rotamers of the i-th residue, from the given starting index for num_rot rotamers
            for rot_i in range(index, index + num_rot):

                # loop over the rotamers of the j-th residue, from the given starting index for num_rot rotamers
                for rot_j in range(index, index + num_rot):


                    # if the current index of one of the two rotamer lists is greater than the number
                    # of rotamers in that list, set the corresponding energy term to zero;
                    # if there is no edge between the two nodes, also set the energy term to zero
                    if (
                        not edge_exists or
                        rot_i >= nrot_i or
                        rot_j >= nrot_j
                    ):
                        energy = 0.0

                    # otherwise store the energy term associated with the edge of the interaction graph
                    # that corresponds to the interaction between the two residue-rotamer choices of the loops
                    else:
                        energy = ig.get_two_body_energy_for_edge(
                            molten_i, molten_j, rot_i, rot_j
                        )

                    data_list.append(
                        {
                            "res i": res_i,
                            "res j": res_j,
                            "rot A_i": rot_i,
                            "rot B_j": rot_j,
                            "E_ij": energy,
                        }
                    )

    df = pd.DataFrame(data_list)
    return df


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
        init(options="-mute all") # no log


    # create the pose from the input PDB and store the default scoring function in a variable
    pose, sfxn = prepare_pose(pdb_path)

    # store the number of residues in a variable
    num_res = pose.total_residue()

    # build the rotamer sets and the interaction graph from the pose and the scoring function
    rotsets, ig = build_rotamers(pose, sfxn)

    # create the file that associates each residue with its number of rotamers
    df_n_rot_per_res = save_num_rotamers_per_residue(
        rotsets, num_res,
    )

    # save the values
    out_path_n_rot = N_ROT_PER_RES / f"nrot_per_res_{num_res}res_manual.csv"
    df_n_rot_per_res.to_csv(out_path_n_rot, index=False)
    print(f"\n\nSaved rotamer count per residue to: {out_path_n_rot}")

    # extract the one-body energy terms
    df_one_body = extract_one_body_data(
        rotsets, ig, num_rot, index, num_res
    )

    # extract the two-body energy terms
    df_two_body = extract_two_body_data(
        rotsets, ig, num_rot, index, num_res
    )

    # save the values
    out_path_ob = ENERGY_FILES / f"{args.num_rot}rot_{num_res}res_one_body_terms_manual.csv"
    df_one_body.to_csv( out_path_ob, index=False)

    out_path_tb = ENERGY_FILES / f"{args.num_rot}rot_{num_res}res_two_body_terms_manual.csv"
    df_two_body.to_csv( out_path_tb, index=False)

    print(f"\n\nEnergy data saved to: {out_path_ob} and {out_path_tb}")


if __name__ == "__main__":
    main()