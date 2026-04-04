from oncoscape.adapter import biomarker_adapter_spec
from oncoscape.cli._common import run_step


def main() -> None:
    run_step("Step 07: run biomarker adapter.", biomarker_adapter_spec)
