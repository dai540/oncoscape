from oncoscape.cli._common import run_step
from oncoscape.training import train_model_plan


def main() -> None:
    run_step("Step 05: train the compartment-first model.", train_model_plan)
