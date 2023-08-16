import os


def get_cfgs_name_path():
    path = os.path.dirname(__file__)
    mapping = {}
    for root, dirs, files in os.walk(path):
        for file_ in files:
            if file_.endswith('.py') and not file_.startswith(
                    '.') and not file_.startswith('_'):
                mapping[file_[:-3]] = os.path.join(root, file_)
    return mapping


cfgs_name_path = get_cfgs_name_path()

__all__ = ['cfgs_name_path']
