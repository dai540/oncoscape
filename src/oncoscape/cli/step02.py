from oncoscape.cli._common import run_step
from oncoscape.reference import build_reference_plan


def main() -> None:
    run_step("Step 02: build a breast reference atlas.", build_reference_plan)
