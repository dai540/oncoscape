from oncoscape.cli._common import run_cli
from oncoscape.reporting import generate_reports


def main() -> None:
    run_cli("Generate stakeholder-facing reports.", generate_reports)
