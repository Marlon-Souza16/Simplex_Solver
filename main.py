import pandas as pd
import numpy as np
from simplex_solver import SimplexSolver


def _prep_simplex(c, A, b, maximize=True):
    return SimplexSolver(np.asarray(c, float),
                         np.asarray(A, float),
                         np.asarray(b, float),
                         maximize=maximize)

def load_lp_excel(path, maximize=True,
                  sheet_obj='objective',
                  sheet_cons='constraints'):
    c  = pd.read_excel(path, sheet_name=sheet_obj,  header=None).iloc[0]
    df = pd.read_excel(path, sheet_name=sheet_cons, header=None)
    A, b = df.iloc[:, :-1], df.iloc[:, -1]
    return _prep_simplex(c, A, b, maximize)

def load_lp_csv(obj_path, cons_path, maximize=True):
    """Lê dois .csv: um para c e outro para A|b."""
    c  = pd.read_csv(obj_path, header=None).iloc[0]
    df = pd.read_csv(cons_path, header=None)
    A, b = df.iloc[:, :-1], df.iloc[:, -1]
    return _prep_simplex(c, A, b, maximize)

def load_lp_single_csv(path, maximize=True):
    """
    Alternativa: tudo num único CSV (long-format):

        type,a1,a2,...,an,rhs
        obj ,4 ,3 ,7 ,
        con ,2 ,1 ,1 ,8
        ...

    """
    df = pd.read_csv(path)
    obj_row = df[df['type']=='obj'].iloc[0, 1:-1]
    cons    = df[df['type']=='con'].reset_index(drop=True)
    A, b = cons.iloc[:, 1:-1], cons['rhs']
    return _prep_simplex(obj_row, A, b, maximize)


solver = load_lp_excel()
status, z_opt, x_opt = solver.solve(verbose=True)
print('Ótimo?', status, '| z* =', z_opt, '| x* =', x_opt)
solver.plot_objective_progress()
