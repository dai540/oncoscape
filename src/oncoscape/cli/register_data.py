from oncoscape.cli._common import run_cli
from oncoscape.data import build_registry


def main() -> None:
    run_cli("Register public breast data sources.", build_registry)
