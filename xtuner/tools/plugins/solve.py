# Copyright (c) OpenMMLab. All rights reserved.
import math
import re
from math import *  # noqa: F401, F403

from sympy import Eq, solve, symbols

from .calculate import Calculate


def Solve(equations_str):
    try:
        equations_str = equations_str.replace(' ', '')
        equations_ori = re.split(r'[,;]+', equations_str)
        equations_str = equations_str.replace('^', '**')
        equations_str = re.sub(r'(\(.*\))([a-zA-Z])', r'\1 * \2',
                               equations_str)
        equations_str = re.sub(r'(\d+)([a-zA-Z])', r'\1 * \2', equations_str)
        equations_str = equations_str.replace('pi', str(math.pi))
        equations = re.split(r'[,;]+', equations_str)
        vars_list = list(set(re.findall(r'[a-zA-Z]+', equations_str)))
        vars = {var: symbols(var) for var in vars_list}

        output = ''
        eqs = []
        for eq in equations:
            if '=' in eq:
                left, right = eq.split('=')
                eqs.append(
                    Eq(
                        eval(left.strip(), {}, vars),
                        eval(right.strip(), {}, vars)))
        solutions = solve(eqs, vars, dict=True)

        vars_values = {var: [] for var in vars_list}
        if isinstance(solutions, list):
            for idx, solution in enumerate(solutions):
                for var, sol in solution.items():
                    output += f'{var}_{idx} = {sol}\n'
                    vars_values[str(var)].append(sol)
        else:
            for var, sol in solutions.items():
                output += f'{var} = {sol}\n'
                vars_values[str(var)].append(sol)
        for eq, eq_o in zip(equations, equations_ori):
            if '=' not in eq:
                for var in vars_list:
                    need_note = True if len(vars_values[var]) > 1 else False
                    for idx, value in enumerate(vars_values[var]):
                        eq_to_calc = eq.replace(var, str(value))
                        calc_result = Calculate(eq_to_calc)
                        if need_note:
                            eq_name = eq_o.replace(var, f'{var}_{idx}')
                        else:
                            eq_name = eq_o
                        if calc_result != 'No results.':
                            output += f'{eq_name} = {calc_result}\n'

        return output.strip()
    except Exception:
        return 'No result.'
