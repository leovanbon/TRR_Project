import random
import pandas as pd
from pysat.solvers import Solver
from pysat.formula import CNF

def generate_sr_unsat(n_vars, k_mean=4):
    formula = CNF()
    with Solver(name='g3') as solver:
        while True:
            k = max(1, min(n_vars, int(random.gauss(k_mean, 1))))
            vars_in_clause = random.sample(range(1, n_vars + 1), k)
            clause = [v if random.random() < 0.5 else -v for v in vars_in_clause]
            solver.add_clause(clause)
            formula.append(clause)

            if not solver.solve():
                return formula.clauses

def to_dimacs(clauses, n_vars):
    dimacs = [f"p cnf {n_vars} {len(clauses)}"]
    for clause in clauses:
        dimacs.append(" ".join(map(str, clause)) + " 0")
    return "\\n".join(dimacs)

def save_to_csv(num_formulas, var_range=(10, 40), out_file="data/my_dataset/train.csv"):
    formulas, sats, proofs = [], [], []

    for _ in range(num_formulas):
        n_vars = random.randint(*var_range)
        clauses = generate_sr_unsat(n_vars)
        dimacs_str = to_dimacs(clauses, n_vars)
        formulas.append(dimacs_str)
        sats.append(False)
        proofs.append("")  # You can later populate this with actual resolution proofs

    df = pd.DataFrame({
        "formula": formulas,
        "sat": sats,
        "res_proof": proofs,
    })
    df.to_csv(out_file, index=False)
    print(f"Saved {num_formulas} UNSAT formulas to {out_file}")

# Run this to generate your dataset
if __name__ == "__main__":
    save_to_csv(num_formulas=1000, var_range=(10, 40), out_file="data/my_dataset/train.csv")
