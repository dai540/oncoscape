from oncoscape.cli._common import run_cli
from oncoscape.training import train_breast_model


def main() -> None:
    run_cli("Train the breast H&E spatial biology model.", train_breast_model)
