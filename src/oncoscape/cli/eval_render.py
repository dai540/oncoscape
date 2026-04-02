from oncoscape.cli._common import run_cli
from oncoscape.evaluation import evaluate_and_render


def main() -> None:
    run_cli("Evaluate the trained model and render outputs.", evaluate_and_render)
