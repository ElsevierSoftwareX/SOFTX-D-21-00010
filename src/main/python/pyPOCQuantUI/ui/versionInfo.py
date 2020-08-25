from pathlib import Path


def get_version_string() -> str:
    """Return the version string in the form VERSION.BUILD."""

    # Get the folder
    folder = Path(__file__).parents[0]

    # Get the version
    version = "0.0.0"
    with open(folder / 'VERSION', 'r') as f:
        line = f.readline()
        line = line.strip()
        version = line

    # Get the build number
    build = 0
    if (folder / 'BUILD').is_file():
        with open(folder / 'BUILD', 'r') as f:
            line = f.readline()
            line = line.strip()
            build = line
    else:
        reset_build_file()

    # Return
    return f"{version}.{build}"


def reset_build_file():
    """Resets build to 0 in the BUILD file."""

    # Get the folder
    folder = Path(__file__).parents[0]

    # Write build 0
    with open(folder / 'BUILD', 'w') as f:
        f.write("0")


def increase_build_number():
    """Increase current build number."""

    # Get the folder
    folder = Path(__file__).parents[0]

    # Create the BUILD file if it does not exist
    if not (folder / 'BUILD').is_file():
        reset_build_file()

    # Read the build number, increase it and save it
    with open(folder / 'BUILD', 'r') as f:
        line = f.readline()
        line = line.strip()
        build = int(line)

    build += 1
    with open(folder / 'BUILD', 'w') as f:
        f.write(str(build))


def compare_version(version: str) -> bool:
    """Compares provided 'version' string with current version in VERSION file.

    The BUILD number is ignored.
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
