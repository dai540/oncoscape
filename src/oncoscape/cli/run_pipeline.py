from oncoscape.cli._common import run_cli
from oncoscape.pipeline import run_pipeline


def main() -> None:
    run_cli("Run the complete oncoscape breast pipeline.", run_pipeline)
