from oncoscape.cli._common import run_cli
from oncoscape.preprocessing import extract_patches_and_graphs


def main() -> None:
    run_cli("Extract patches and spatial graphs.", extract_patches_and_graphs)
