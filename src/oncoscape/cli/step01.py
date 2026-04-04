from oncoscape.cli._common import run_step
from oncoscape.data import register_data_plan


def main() -> None:
    run_step("Step 01: register public breast data.", register_data_plan)
