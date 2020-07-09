import ast


def load_settings(filename):
    """Loads settings from file and returns them in a dictionary."""
    settings_dictionary = {}
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        key, value = line.split("=")
        settings_dictionary[key.strip()] = ast.literal_eval(value.strip())
    return settings_dictionary


def save_settings(settings_dictionary, filename):
    """Save settings from a dictionary to file."""
    with open(filename, "w+") as f:
        for key in settings_dictionary:
            f.write(f"{key}={settings_dictionary[key]}\n")
