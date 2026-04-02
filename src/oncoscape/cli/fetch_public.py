from oncoscape.cli._common import run_cli
from oncoscape.data import fetch_public_data


def main() -> None:
    run_cli("Fetch public data files defined in the oncoscape download catalog.", fetch_public_data)
