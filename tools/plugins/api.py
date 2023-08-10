# Copyright (c) OpenMMLab. All rights reserved.
import re


def plugins_api(input_str,
                calculate_open=True,
                solve_open=True,
                search_open=True):

    pattern = r'(Solve|solve|Solver|solver|Calculate|calculate|Calculator|calculator|Search)\("([^"]*)"\)'  # noqa: E501

    matches = re.findall(pattern, input_str)

    converted_str = '<|Results|>:\n'

    for i in range(len(matches)):
        if matches[i][0] in [
                'Calculate', 'calculate'
                'Calculator', 'calculator'
        ]:
            if calculate_open:
                from .calculate import Calculate
                result = Calculate(matches[i][1])
            else:
                result = None
            converted_str += f"Calculate(\"{matches[i][1]}\") => {result}\n"
        elif matches[i][0] in ['Solve', 'solve', 'Solver', 'solver']:
            if solve_open:
                from .solve import Solve
                result = Solve(matches[i][1])
            else:
                result = None
            converted_str += f"Solve(\"{matches[i][1]}\") =>\n{result}\n"
        elif matches[i][0] == 'Search':
            if search_open:
                from .search import Search
                result = Search(matches[i][1])
            else:
                result = None
            converted_str += f"Search(\"{matches[i][1]}\") =>\n{result}"

    converted_str += '<eor>\n'
    return converted_str
