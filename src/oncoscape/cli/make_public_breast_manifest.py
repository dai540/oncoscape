from oncoscape.cli._common import run_cli
from oncoscape.data import build_public_breast_manifest


def main() -> None:
    run_cli("Assemble a breast public-data source manifest from downloaded datasets.", build_public_breast_manifest)
