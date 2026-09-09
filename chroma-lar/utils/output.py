from rich.console import Console
from rich.table import Table

import h5py
import os
from .log import logger
from typing import List

class H5Logger:
    def __init__(self, filename: str, variables: List[str]):
        self.filename = filename
        self.variables = variables

        if os.path.exists(filename):
            logger.warning(f"File {filename} already exists. Overwriting.")
            os.remove(filename)
        self.f = h5py.File(filename, "a")

        for var in self.variables:
            if var not in self.f:
                self.f.create_dataset(var, (0,), maxshape=(None,), dtype="f")

    def write(self, **kwargs):
        for var in self.variables:
            data = self.f[var]
            data.resize((data.shape[0] + 1,))
            data[-1] = kwargs[var]

    def close(self):
        self.f.close()

def print_table(**kwargs):
    """Print a summary table of the simulation"""
    console = Console()
    table = Table(title="Chroma-LXE Simulation Summary")
    table.add_column("metric", style="bold red")
    table.add_column("value", style="bold red")
    for key, value in kwargs.items():
        table.add_row(key, str(value))
    console.print(table)