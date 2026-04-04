from oncoscape.cli._common import run_step
from oncoscape.plans import fetch_and_manifest_plan


def main() -> None:
    run_step("Step 00: fetch public breast data and build a manifest.", fetch_and_manifest_plan)
