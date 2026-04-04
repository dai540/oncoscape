from oncoscape.cli._common import run_step
from oncoscape.preprocessing import extract_tiles_plan


def main() -> None:
    run_step("Step 04: extract tiles and graphs.", extract_tiles_plan)
