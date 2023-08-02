import re

from .calculate import Calculate
from .search import Search
from .solve import Solve


def plugins_api(input_str):

    pattern = r'(Solve|solve|Solver|solver|Calculate|calculate|Calculator|calculator|Search)\("([^"]*)"\)'  # noqa: E501

    matches = re.findall(pattern, input_str)

    converted_str = '<|Results|>:\n'

    for i in range(len(matches)):
        if matches[i][0] in [
                'Calculate', 'calculate'
                'Calculator', 'calculator'
        ]:
            result = Calculate(matches[i][1])
            converted_str += f"Calculate(\"{matches[i][1]}\") => {result}\n"
        elif matches[i][0] in ['Solve', 'solve', 'Solver', 'solver']:
            result = Solve(matches[i][1])
            converted_str += f"Solve(\"{matches[i][1]}\") =>\n{result}\n"
        elif matches[i][0] == 'Search':
            result = Search(matches[i][1])
            converted_str += f"Search(\"{matches[i][1]}\") =>\n{result}"

    converted_str += '<eor>\n'
    return converted_str
