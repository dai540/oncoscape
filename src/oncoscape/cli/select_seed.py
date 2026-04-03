from oncoscape.cli._common import run_cli
from oncoscape.reporting.selection import summarize_seed_sweep


def main() -> None:
    run_cli("Summarize a seed sweep and promote the best validation seed.", summarize_seed_sweep)
