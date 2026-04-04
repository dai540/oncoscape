from oncoscape.cli._common import run_step
from oncoscape.plans import biomarker_adapter_spec


def main() -> None:
    run_step("Step 07: run biomarker adapter.", biomarker_adapter_spec)
