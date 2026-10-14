import argparse
import pandas as pd
from gurobipy import Model, GRB, quicksum

from .hybrid_annealing import get_Q, format_binary_string, from_sample_to_energy

from .config import N_ROT_PER_RES, DATA_DIR

## To run: python -m scripts.gurobi -res <num_res> -rot <num_rot> [-s <seed>]

## Time limit: set TIME_LIMIT below (seconds)
TIME_LIMIT = 10  # seconds

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
    parser.add_argument("-s",
        "--seed",
        type=int,
        default=0)
    return parser.parse_args()

def main():

    # store user input in variables
    args = parse_args()
    num_res = args.num_res
    num_rot = args.num_rot
    num_var = num_res * num_rot

    # define the model
    m = Model("protein_folding_exact")
    m.setParam("OutputFlag", 0)    # suppress solver logs
 
    # MIPFocus=1: the solver prioritizes finding good-quality feasible solutions
    # quickly rather than proving optimality (default: MIPFocus=0)
    m.setParam("MIPFocus", 1)
 
    # Heuristics: fraction of computation time spent on heuristics searching for
    # feasible solutions instead of exact branch-and-bound (default 0.05)
    m.setParam("Heuristics", 0.5)
 
    # MIPGap: relative gap threshold below which the solver stops (default 1e-4)
    m.setParam("MIPGap", 0.1)

    # Seed so that repeated runs differ from each other
    m.setParam("Seed", args.seed)

    # define variables
    x = m.addVars(range(num_var), vtype=GRB.BINARY, name="x")

    # constraint: variables associated with non-existent residue-rotamer pairs must be set to zero
    # load file with the actual number of rotamers per residue
    rotamer_count_file_path = N_ROT_PER_RES / f"nrot_per_res_{num_res}res.csv"
    if not rotamer_count_file_path.exists():
        rotamer_count_file_path = N_ROT_PER_RES / f"nrot_per_res_{num_res}res_manual.csv"    
    df = pd.read_csv(rotamer_count_file_path)

    # build list of rotamer counts for each residue
    nrot_per_res_list = df.sort_values("residue")["num_rotamers"].astype(int).tolist()

    # check and add constraints if the actual number of rotamers is less than num_rot
    for res in range(num_res):
        nrot_real = nrot_per_res_list[res]  # actual number of rotamers for this residue
        if nrot_real < num_rot:
            # add x[idx] == 0 constraints for all rotamers that do not exist
            for r in range(nrot_real, num_rot):
                idx = res * num_rot + r
                constraint = x[idx] == 0
                m.addConstr(constraint)

    # constraint: only one rotamer per residue can be chosen
    for i in range(0, num_var, num_rot):
        m.addConstr(quicksum(x[i + j] for j in range(num_rot)) == 1)

    # define and set the objective function
    Q = get_Q(num_res, num_rot)

    obj = quicksum(Q[i, i] * x[i] for i in range(num_var))
    for i in range(num_var):
        for j in range(i + 1, num_var):
            if Q[i, j] != 0:
                obj += Q[i, j] * x[i] * x[j]

    m.setObjective(obj, GRB.MINIMIZE)
    
    # set time_limit
    m.setParam('TimeLimit', TIME_LIMIT)

    # solve
    m.optimize()
    
    # extract solution
    sol = m.getAttr("X", x)
    solution_bits = ''.join(str(int(round(sol[i]))) for i in range(num_var))

    # extract energy associated with the solution
    gurobi_energy = m.ObjVal

    # extract runtime
    gurobi_runtime = getattr(m, "Runtime", None)

    # extract status
    status_code = m.status
    # map of the main status names for a readable CSV
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.NUMERIC: "NUMERIC_ISSUE",
    }
    status_name = status_map.get(status_code, str(status_code))

    # extract MIPGap
    mip_gap = float(m.MIPGap)

    # save results
    row = {
        'num_res': int(num_res),
        'num_rot': int(num_rot),
        'gs_bitstring': format_binary_string(solution_bits, num_res, num_rot),
        'gs_energy': gurobi_energy,
        'execution_time': gurobi_runtime,
        'status': status_name,
        'mip_gap': mip_gap if mip_gap is not None else float('nan'),
        'seed': args.seed
    }

    file_output_path = DATA_DIR / "gurobi.csv"

    if file_output_path.exists():
        # new df, then concatenate old df and new df
        df_existing = pd.read_csv(file_output_path)
        df_new = pd.DataFrame([row])
        df_to_save = pd.concat([df_existing, df_new], ignore_index=True)
        # to sort the values (otherwise comment out)
        # df_to_save = (df_to_save.sort_values(by=["num_res", "num_rot"], ascending=[True, True], kind="stable").reset_index(drop=True))
        df_to_save.to_csv(file_output_path, index=False)
    else:
        # if it does not exist, create the file
        df_new = pd.DataFrame([row])
        df_new.to_csv(file_output_path, index=False)

    print(f"File updated or created in: {file_output_path}")
    print(f"Status: {status_name}, mip_gap: {mip_gap}, runtime: {gurobi_runtime}")


    # ## prints: ##
    
    # print("\nQ:\n")

    # labels = [f"x_{i}_{j}" for i in range(num_res) for j in range(num_rot)]
    # df_Q = pd.DataFrame(Q, index=labels, columns=labels)
    # print(df_Q.round(6).to_string())

    # print("\n Objective:\n")

    # print(obj)

    # print("\n gs_bitstring:\n")

    # print(format_binary_string(solution_bits, num_res, num_rot))

    # print("\n gs_energy:\n")

    # print(gurobi_energy)

    # print("\n Energy associated with gs_bitstring:\n")

    # print(from_sample_to_energy(sol, Q))

    # #############


if __name__ == "__main__":
    main()