import multiprocessing

params = [
    {
        'name': 'Runtime parameters', 'type': 'group', 'children':
        [
            {'name': 'Number of cores (max={})'.format(multiprocessing.cpu_count()), 'type': 'int', 'value': 2,
             'limits': (1, multiprocessing.cpu_count()), 'default': 1, 'tip': "Number of cores to use for processing"},
            {'name': 'QC', 'type': 'bool', 'value': True, 'tip': "Save quality control images"},
            {'name': 'Verbose', 'type': 'bool', 'value': True, 'tip': "Print useful information"}
        ]
    },
    {
        'name': 'Basic parameters', 'type': 'group', 'children':
        [
            {'name': 'Sensor band names', 'type': 'group', 'children':
                [
                    {'name': 't2', 'type': 'str', 'value': 'igm'},
                    {'name': 't1', 'type': 'str', 'value': 'igg'},
                    {'name': 'ctl', 'type': 'str', 'value': 'ctl'}
                ]
             },
            {'name': 'Band expected relative location', 'type': 'group', 'children':
                [
                    {'name': 't2', 'type': 'float', 'value': 0.25, 'step': 0.05, 'limits': (0, 1)},
                    {'name': 't1', 'type': 'float', 'value': 0.53, 'step': 0.05, 'limits': (0, 1)},
                    {'name': 'ctl', 'type': 'float', 'value': 0.79, 'step': 0.05, 'limits': (0, 1)},
                ]
             },
            {'name': 'Sensor center', 'type': 'group', 'children':
                [
                    {'name': 'y', 'type': 'int', 'value': 10},
                    {'name': 'x', 'type': 'int', 'value': 10}
                ]
            },
            {'name': 'Sensor size', 'type': 'group', 'children':
                [
                    {'name': 'height', 'type': 'int', 'value': 10},
                    {'name': 'width', 'type': 'int', 'value': 10}
                ]
            },
            {'name': 'Sensor border', 'type': 'group', 'children':
                [
                    {'name': 'y', 'type': 'int', 'value': 7},
                    {'name': 'x', 'type': 'int', 'value': 7},
                ]
            },
            {'name': 'Perform sensor search', 'type': 'bool', 'value': True, 'tip': "Search sensor in box"},
            {'name': 'QR code border', 'type': 'int', 'value': 40},
            {'name': 'Subtract background', 'type': 'bool', 'value': True, 'tip': "Subtract background from signal"}
        ]},
    {
        'name': 'Advanced parameters', 'type': 'group', 'children':
        [
            {'name': 'Sensor search area', 'type': 'group', 'children':
                [
                    {'name': 'y', 'type': 'int', 'value': 7},
                    {'name': 'x', 'type': 'int', 'value': 7},
                ]
             },
            {'name': 'Sensor threshold factor', 'type': 'float', 'value': 2.0, 'step': 0.1},
            {'name': 'Raw auto stretch', 'type': 'bool', 'value': False,
             'tip': "Set to true if raw image intensities should be auto-stretched"},
            {'name': 'Raw auto wb', 'type': 'bool', 'value': False,
             'tip': "Set true if raw image white balance should be set automatically"},
            {'name': 'Try to correct strip orientation', 'type': 'bool', 'value': True,
             'tip': "Will try to find the injection inlet to assess orientation."},
            {'name': 'Strip orientation correction search rectangles', 'type': 'group', 'children':
                [
                    {'name': 'Relative height factor', 'type': 'float', 'value': 0.52, 'step': 0.01, 'limits': (0, 1)},
                    {'name': 'Relative center cut-off', 'type': 'float', 'value': 0.15, 'step': 0.01, 'limits': (0, 1)},
                    {'name': 'Relative border cut-off', 'type': 'float', 'value': 0.09, 'step': 0.01, 'limits': (0, 1)},
                ]
             },
            {'name': 'Strip text to search (orientation)', 'type': 'str', 'value': '', 'default': '',
             'tip': "Use prominent text such as COVID"},
            {'name': 'Strip text is on the right', 'type': 'bool', 'value': False,
             'tip': "Set true if test is right of the sensor"},
            {'name': 'Force FID search', 'type': 'bool', 'value': False,
             'tip': "Set true to force FID search if the QR code with patient data is not present"}
        ]
    }
]

key_map = {
    'raw_auto_stretch': 'raw_auto_stretch',
    'raw_auto_wb': 'raw_auto_wb',
    'try_to_correct_strip_orientation': 'strip_try_correct_orientation',
    'strip_orientation_correction_search_rectangles': 'strip_try_correct_orientation_rects',
    'strip_text_to_search_(orientation)': 'strip_text_to_search',
    'strip_text_is_on_the_right': 'strip_text_on_right',
    'poct_size': 'strip_size',
    'sensor_center': 'sensor_center',
    'sensor_size': 'sensor_size',
    'sensor_border': 'sensor_border',
    'perform_sensor_search': 'perform_sensor_search',
    'sensor_search_area': 'sensor_search_area',
    'sensor_threshold_factor': 'sensor_thresh_factor',
    'qr_code_border': 'qr_code_border',
    'subtract_background': 'subtract_background',
    'band_expected_relative_location': 'peak_expected_relative_location',
    'force_fid_search': 'force_fid_search',
    'sensor_band_names': 'sensor_band_names',
    'number_of_cores_(max={})'.format(multiprocessing.cpu_count()): 'max_workers',
    'qc': 'qc',
    'verbose': 'verbose'
}
