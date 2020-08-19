import multiprocessing
import ast


params = [
    {'name': 'Basic parameters', 'type': 'group', 'children': [
        {'name': 'Raw auto stretch', 'type': 'bool', 'value': False, 'tip': "Set true if raw image should be auto "
                                                                            "streched"},
        {'name': 'Raw auto wb', 'type': 'bool', 'value': False, 'tip': "Set true if raw image' white balance should be"
                                                                       " set automatically"},
        {'name': 'Strip text to search (orientation)', 'type': 'str', 'value': '""',  'default': '""',
         'tip': "Use prominent text such as COVID"},
        {'name': 'Strip text is on the right', 'type': 'bool', 'value': False, 'tip': "Set true if test is right of the"
                                                                                      " sensor"},
        {'name': 'POCT size', 'type': 'group', 'children': [
            {'name': 'width', 'type': 'int', 'value': 10},
            {'name': 'height', 'type': 'int', 'value': 10},
        ]},
        {'name': 'Sensor center', 'type': 'group', 'children': [
            {'name': 'x', 'type': 'int', 'value': 10},
            {'name': 'y', 'type': 'int', 'value': 10},
        ]},
        {'name': 'Sensor size', 'type': 'group', 'children': [
            {'name': 'width', 'type': 'int', 'value': 10},
            {'name': 'height', 'type': 'int', 'value': 10},
        ]},
        {'name': 'Sensor border', 'type': 'group', 'children': [
            {'name': 'x', 'type': 'int', 'value': 7},
            {'name': 'y', 'type': 'int', 'value': 7},
        ]},
        {'name': 'Perform sensor search', 'type': 'bool', 'value': True, 'tip': "Search sensor in box"},
        {'name': 'Sensor search area', 'type': 'group', 'children': [
            {'name': 'x', 'type': 'int', 'value': 7},
            {'name': 'y', 'type': 'int', 'value': 7},
        ]},
        {'name': 'Sensor threshold factor', 'type': 'int', 'value': 2},
        {'name': 'Min sensor score', 'type': 'float', 'value': .85, 'step': 0.01},
        {'name': 'QR code border', 'type': 'int', 'value': 40},
        {'name': 'Subtract background', 'type': 'bool', 'value': True, 'tip': "Subtract background from signal"},
        {'name': 'Peak expected relative location', 'type': 'group', 'children': [
            {'name': 'IgM', 'type': 'float', 'value': 0.25},
            {'name': 'IgG', 'type': 'float', 'value': 0.53},
            {'name': 'Ctl', 'type': 'float', 'value': 0.79},
        ]},
        {'name': 'Number of cores (min=1;max={})'.format(multiprocessing.cpu_count()), 'type': 'int', 'value': 2,
         'limits': (1, multiprocessing.cpu_count()), 'default': 1, 'tip': "Number of cores to use for processing"},
        {'name': 'QC', 'type': 'bool', 'value': True, 'tip': "Save quality control images"},
        {'name': 'Verbose', 'type': 'bool', 'value': True, 'tip': "Print useful information"},
        {'name': 'File version', 'type': 'int', 'value': 1},
    ]}]

key_map = {
    'raw_auto_stretch': 'raw_auto_stretch',
    'raw_auto_wb': 'raw_auto_wb',
    'strip_text_to_search_(orientation)': 'strip_text_to_search',
    'strip_text_is_on_the_right': 'strip_text_on_right',
    'poct_size': 'strip_size',
    'sensor_center': 'sensor_center',
    'sensor_size': 'sensor_size',
    'sensor_border': 'sensor_border',
    'perform_sensor_search': 'perform_sensor_search',
    'sensor_search_area': 'sensor_search_area',
    'sensor_threshold_factor': 'sensor_thresh_factor',
    'min_sensor_score': 'min_sensor_score',
    'qr_code_border': 'qr_code_border',
    'subtract_background': 'subtract_background',
    'peak_expected_relative_location': 'peak_expected_relative_location',
    'number_of_cores_(min=1;max={})'.format(multiprocessing.cpu_count()): 'max_workers',
    'qc': 'qc',
    'verbose': 'verbose',
    'file_version': 'file_version'
}


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