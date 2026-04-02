from oncoscape.cli._common import run_cli
from oncoscape.labels import build_teachers


def main() -> None:
    run_cli("Build pathology and spatial teachers.", build_teachers)
