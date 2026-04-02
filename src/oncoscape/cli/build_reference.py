from oncoscape.cli._common import run_cli
from oncoscape.reference import build_reference_atlas


def main() -> None:
    run_cli("Build a breast scRNA reference atlas.", build_reference_atlas)
