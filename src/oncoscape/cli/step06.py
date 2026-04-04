from oncoscape.cli._common import run_step
from oncoscape.plans import evaluate_plan


def main() -> None:
    run_step("Step 06: evaluate and render maps.", evaluate_plan)
