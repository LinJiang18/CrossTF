from pprint import pprint
from typing import Any


class PrintLogger:
    """Minimal console logger used by the training loop."""

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def log(self, name: str, data: Any, step=None):
        print(f'{name}: {data}' if step is None else f'step {step}, {name}: {data:.4e}')

    def log_name_params(self, name: str, params: Any):
        print(f'{name}:')
        pprint(params)
