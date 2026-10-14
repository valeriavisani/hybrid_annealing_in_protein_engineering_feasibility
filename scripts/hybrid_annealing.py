import argparse
import numpy as np
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from dimod import Binary, ConstrainedQuadraticModel

from .config import ENERGY_FILES, N_ROT_PER_RES, DATA_DIR
from .configto import DWAVE_TOKEN

## To run: python -m scripts.hybrid_annealing -res <num_res> -rot <num_rot> [-run_id <run_id>]

def parse_args():
    # function to read user input

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rot",
        "--num_rot",
        type=int,
        required=True,
        help="Number of rotamers per residue"
    )
    parser.add_argument(
        "-res",
        "--num_res",
        type=int,
        required=True,
        help="Number of residues"
    )
    parser.add_argument("-run_id",
        "--run_id",
        type=int,
        default=0)
    return parser.parse_args()


def get_Q(num_res: int, num_rot: int) -> np.ndarray:
    # function that returns Q given num_res and num_rot

    # load energy file paths
    one_body_file_path = ENERGY_FILES / f"{num_rot}rot_{num_res}res_one_body_terms.csv"
    if not one_body_file_path.exists():
        one_body_file_path = ENERGY_FILES / f"{num_rot}rot_{num_res}res_one_body_terms_manual.csv"
    
    two_body_file_path = ENERGY_FILES / f"{num_rot}rot_{num_res}res_two_body_terms.csv"
    if not two_body_file_path.exists():
        two_body_file_path = ENERGY_FILES / f"{num_rot}rot_{num_res}res_two_body_terms_manual.csv"

    # load one-body terms
    one_body_df = pd.read_csv(one_body_file_path)
    q = one_body_df["E_ii"].values
    dim = len(q)

    # load two-body terms
    two_body_df = pd.read_csv(two_body_file_path)
    values = two_body_df["E_ij"].values

    # initialize Q
    Q = np.zeros((dim, dim))

    # insert one-body terms into Q
    Q[np.arange(dim), np.arange(dim)] = q

    # insert two-body terms into Q
    idx = 0
    num_var = num_rot * num_res
    for start_row in range(0, num_var, num_rot):
        for start_col in range(start_row + num_rot, num_var, num_rot):
            for i in range(num_rot):
                for j in range(num_rot):
                    Q[start_row + i, start_col + j] = values[idx]
                    Q[start_col + j, start_row + i] = values[idx]
                    idx += 1

    return Q

def define_cqm(num_res, num_rot, Q):
    # function that returns the constrained quadratic model given num_res, num_rot and Q

    # define the variables
    num_var = num_res * num_rot
    x = {n: Binary(n) for n in range(num_var)}

    # define the constrained quadratic model
    cqm = ConstrainedQuadraticModel()
    
    # constraint: variables associated with non-existent residue-rotamer pairs must be set to zero
    nrot_per_res_list = load_nrot_per_res(num_res)

    # check and add constraints if the actual number of rotamers is less than num_rot
    for res in range(num_res):
        nrot_real = nrot_per_res_list[res]  # actual number of rotamers for this residue
        if nrot_real < num_rot:
            # add x[idx] == 0 constraints for all rotamers that do not exist
            for r in range(nrot_real, num_rot):
                idx = res * num_rot + r
                constraint = x[idx] == 0
                cqm.add_constraint(constraint)

    # constraint: only one rotamer per residue can be chosen
    for i in range(0, num_var, num_rot):
        constraint = sum(x[i+j] for j in range(0, num_rot)) == 1
        cqm.add_constraint(constraint)

    # define and set the objective function
    objective = 0

    for i in range(num_var):
        if Q[i, i] != 0:
            objective += Q[i, i] * x[i]

    for i in range(num_var):
        for j in range(i+1, num_var):
            if Q[i, j] != 0:
                objective += Q[i, j] * x[i] * x[j]

    cqm.set_objective(objective)

    return(cqm)

def load_nrot_per_res(num_res):
    # function that returns a list mapping each residue to its actual number of rotamers, given num_res

    # load file with the actual number of rotamers per residue
    rotamer_count_file_path = N_ROT_PER_RES / f"nrot_per_res_{num_res}res.csv"
    if not rotamer_count_file_path.exists():
        rotamer_count_file_path = N_ROT_PER_RES / f"nrot_per_res_{num_res}res_manual.csv"
    
    df = pd.read_csv(rotamer_count_file_path)

    # build list of rotamer counts for each residue
    nrot_per_res_list = df.sort_values("residue")["num_rotamers"].astype(int).tolist()

    return nrot_per_res_list

def da_microsecondi_a_secondi(us):
    # function to convert microseconds to seconds

    return us * 1e-6

def format_binary_string(stringa: str, num_res: int, num_rot: int) -> str:
    # function to format a bitstring

    num_var = num_res * num_rot

    if len(stringa) != num_var:
        raise ValueError(f"The string length ({len(stringa)}) must be equal to num_res * num_rot ({num_var}).")

    blocks = [stringa[i:i + num_rot] for i in range(0, num_var, num_rot)]

    return "   ".join(blocks)

def from_sample_to_energy(sample_arr: np.array, Q: np.array) -> float:
    # function to compute the energy associated with a bitstring, given the bitstring and Q
    # only used in the commented-out prints, useful for checking

    lenght = len(sample_arr)

    E = 0.0
    for i in range(lenght):
        E += Q[i, i] * sample_arr[i]
        for j in range(i+1, lenght):
            if Q[i, j] != 0:
                E += Q[i, j] * sample_arr[i] * sample_arr[j]
    
    return E

def main():
    
    # store user input in variables
    args = parse_args()
    num_res = args.num_res
    num_rot = args.num_rot
    num_var = num_res * num_rot

    # define Q
    Q = get_Q(num_res, num_rot)

    # define the model
    cqm = define_cqm(num_res, num_rot, Q)

    # store token
    # token = DWAVE_TOKEN

    # define the sampler
    sampler = LeapHybridCQMSampler(token=token) 

    # define the time limit
    execution_time = 5

    ## prints: ##

    # Q = get_Q(num_res, num_rot)
    # labels = [f"x_{i}_{j}" for i in range(num_res) for j in range(num_rot)]
    # df_Q = pd.DataFrame(Q, index=labels, columns=labels)
    # print(df_Q.round(6).to_string())

    #############

    # the solver runs the computation and returns the sampleset
    sampleset = sampler.sample_cqm(cqm, 
                                time_limit=execution_time,  # seconds
                                label='Protein Folding')
    
    # filter the feasible solutions
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    
    if len(feasible_sampleset) == 0:
        print("No feasible solution found")
        return

    # extract solution
    best_solution = feasible_sampleset.first.sample

    print("\nBest solution found:")
    solution_bits = ''.join(str(int(round(best_solution[i]))) for i in range(num_var))
    solution_bits_formatted = format_binary_string(solution_bits, num_res, num_rot)
    print(solution_bits_formatted)

    # extract energy associated with the solution
    energy = feasible_sampleset.first.energy
    print(f"\nBest solution - energy = {energy:.6f}")
    
    # extract execution time information
    timing_info = feasible_sampleset.info

    run_time = timing_info["run_time"]
    qpu_access_time = timing_info["qpu_access_time"]

    print("QPU Access Time (μs):", qpu_access_time)
    print("Hybrid Solver Runtime (μs):", run_time)

    # save results
    row = {
        "num_res": int(num_res),
        "num_rot": int(num_rot),
        "gs_bitstring": solution_bits_formatted,
        "gs_energy [REU]": energy,
        "run_time [s]": da_microsecondi_a_secondi(run_time),
        "QPU_access_time [s]": da_microsecondi_a_secondi(qpu_access_time)
    }

    file_output_path = DATA_DIR / "hybrid_annealing.csv"

    if file_output_path.exists():
        # new df, concatenate old and new df
        df_existing = pd.read_csv(file_output_path)
        df_new = pd.DataFrame([row])
        df_to_save = pd.concat([df_existing, df_new], ignore_index=True)
        # to sort the values (otherwise comment out)
        # df_to_save = df_to_save.sort_values(by="num_res", ascending=True, kind="stable").reset_index(drop=True)
        df_to_save.to_csv(file_output_path, index=False)
    else:
        # if it does not exist, create the file
        df_new = pd.DataFrame([row])
        df_new.to_csv(file_output_path, index=False)

    print(f"File updated or created at: {file_output_path}")

    # save the full sampleset

    rows = []
    n_solutions = len(feasible_sampleset)

    for i in range(n_solutions):
        sample_arr = feasible_sampleset.record.sample[i] 
        bitstring = ''.join(str(int(round(sample_arr[j]))) for j in range(num_var))
        annealing_energy = float(feasible_sampleset.record.energy[i])

        row = {
            "num_res": num_res,
            "num_rot": num_rot,
            "gs_bitstring": format_binary_string(bitstring, num_res, num_rot),
            "gs_energy [REU]": annealing_energy
        }
        rows.append(row)


    df_sampleset = pd.DataFrame(rows)

    sampleset_path = DATA_DIR / "hybrid_annealing_samplesets" / f"ha_{num_res}res_{num_rot}rot_run{args.run_id}.csv"
    sampleset_path.parent.mkdir(parents=True, exist_ok=True)
    df_sampleset.to_csv(sampleset_path, index=False)
    print(f"Full sampleset saved to: {sampleset_path}")


    # ## prints: ##
    
    # print("\nQ:\n")

    # labels = [f"x_{i}_{j}" for i in range(num_res) for j in range(num_rot)]
    # df_Q = pd.DataFrame(Q, index=labels, columns=labels)
    # print(df_Q.round(6).to_string())

    # print("\n gs_bitstring:\n")

    # print(solution_bits_formatted)

    # print("\n gs_energy:\n")

    # print(energy)

    # print("\n Energy associated with gs_bitstring:\n")

    # print(from_sample_to_energy(best_solution, Q))

    # #############


if __name__ == "__main__":
    main()