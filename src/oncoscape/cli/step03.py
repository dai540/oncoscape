from oncoscape.cli._common import run_step
from oncoscape.plans import build_teacher_plan


def main() -> None:
    run_step("Step 03: build breast-specific teachers.", build_teacher_plan)
