from pathlib import Path


def get_version_string() -> str:
    """Return the version string."""

    # Get the folder
    folder = Path(__file__).parents[0]

    # Get the version
    version = "0.0.0"
    with open(folder / 'VERSION', 'r') as f:
        line = f.readline()
        line = line.strip()
        version = line

    # Return
    return f"{version}"


def compare_version(version: str) -> bool:
    """Compares provided 'version' string with current version in VERSION file.

    :param version: string
        Version string in the form X.Y.Z.
    """

    # Get the folder
    folder = Path(__file__).parents[0]

    # Get the version
    current_version = "0.0.0"
    with open(folder / 'VERSION', 'r') as f:
        line = f.readline()
        line = line.strip()
        current_version = line

    return current_version == version


def set_new_version(version: str):
    """Set new version to VERSION file.

    :param version: string
        New version string in the form X.Y.Z.
    """

    # Get the folder
    folder = Path(__file__).parents[0]

    # Write new version
    with open(folder / 'VERSION', 'w') as f:
        f.write(str(version))
