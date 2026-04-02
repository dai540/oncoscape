from oncoscape.cli._common import run_cli
from oncoscape.validation import run_preflight


def main() -> None:
    run_cli("Run preflight checks for an oncoscape breast pipeline config.", run_preflight)
