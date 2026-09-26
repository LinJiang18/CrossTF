"""Where the solved fusion coefficients live: logs/CrossTF_<dataset>/alpha.json.

The coefficients are solved from several training seeds at once, so the file belongs to
the dataset, not to one seed. It carries the window and the solver settings alongside the
coefficients so that a record can be interpreted on its own.
"""

from __future__ import annotations

import json
import os


def alpha_dir(name: str, log_root: str = './logs') -> str:
    return os.path.join(log_root, f'CrossTF_{name}')


def alpha_path(name: str, log_root: str = './logs') -> str:
    return os.path.join(alpha_dir(name, log_root), 'alpha.json')


def save(result: dict, name: str, log_root: str = './logs', filename: str = 'alpha.json') -> str:
    d = alpha_dir(name, log_root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, filename)
    with open(path, 'w') as f:
        json.dump(result, f, indent=1, default=float)
    return path


def load(name: str, log_root: str = './logs', path: str | None = None) -> dict:
    path = path or alpha_path(name, log_root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'No solved coefficients at {path}. Solve them first:\n'
            f'  python main.py --config_file configs/{name}.yaml --name {name} '
            f'--train --crosstf\n'
            f'or use the released ones: --alpha_file results/CrossTF/{name}/alpha.json'
        )
    with open(path) as f:
        return json.load(f)
