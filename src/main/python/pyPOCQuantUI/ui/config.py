#  ********************************************************************************
#   Copyright © 2020-2021, ETH Zurich, D-BSSE, Andreas P. Cuny & Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the GNU Public License v3.0
#   which accompanies this distribution, and is available at
#   http://www.gnu.org/licenses/gpl
#
#   Contributors:
#     * Andreas P. Cuny - initial API and implementation
#     * Aaron Ponti - initial API and implementation
#  *******************************************************************************

import multiprocessing
from pypocquant.lib.settings import default_settings

defaults = default_settings()

params = [
    {
        'name': 'Runtime parameters',
        'type': 'group',
        'children':
        [
            {
                'name': 'Number of cores (max={})'.format(multiprocessing.cpu_count()),
                'type': 'int',
                'value': 2,
                'limits': (1, multiprocessing.cpu_count()), 'default': 1, 'tip': "Number of cores to use for processing"
            },
            {
                'name': 'QC',
                'type': 'bool',
                'value': defaults['qc'],
                'tip': "Save quality control images"
            },
            {
                'name': 'Verbose',
                'type': 'bool',
                'value': defaults['verbose'],
                'tip': "Print useful information"
            }
        ]
    },
    {
        'name': 'Basic parameters',
        'type': 'group',
        'children':
        [
            {
                'name': 'Number of sensor bands',
                'type': 'int',
                'value': len(defaults['peak_expected_relative_location']),
                'default': len(defaults['peak_expected_relative_location']),
                'limits': (1, 100),
                'tip': "Number of bands on the sensor."
            },
            {
                'name': 'Control band index',
                'type': 'int',
                'value': defaults['control_band_index'],
                'default': defaults['control_band_index'],
                'limits': (-1, 100),
                'tip': "Index of the control band (set to -1 for right-most position)"
            },
            {
                'name': 'Sensor band names',
                'type': 'group',
                'children':
                [
                    {
                        'name': '0',
                        'type': 'str',
                        'value': defaults['sensor_band_names'][0]
                    },
                    {
                        'name': '1',
                        'type': 'str',
                        'value': defaults['sensor_band_names'][1]
                    },
                    {
                        'name': '2',
                        'type': 'str',
                        'value': defaults['sensor_band_names'][2]
                    }
                ]
             },
            {
                'name': 'Band expected relative location',
                'type': 'group',
                'children':
                [
                    {
                        'name': '0',
                        'type': 'float',
                        'value': defaults['peak_expected_relative_location'][0],
                        'step': 0.05,
                        'limits': (0, 1)
                    },
                    {
                        'name': '1',
                        'type': 'float',
                        'value': defaults['peak_expected_relative_location'][1],
                        'step': 0.05,
                        'limits': (0, 1)
                    },
                    {
                        'name': '2',
                        'type': 'float',
                        'value': defaults['peak_expected_relative_location'][2],
                        'step': 0.05,
                        'limits': (0, 1)
                    },
                ]
             },
            {
                'name': 'Sensor center',
                'type': 'group',
                'children':
                [
                    {
                        'name': 'y',
                        'type': 'int',
                        'value': defaults['sensor_center'][0],
                    },
                    {
                        'name': 'x',
                        'type': 'int',
                        'value': defaults['sensor_center'][1]
                    }
                ]
            },
            {
                'name': 'Sensor size',
                'type': 'group',
                'children':
                [
                    {
                        'name': 'height',
                        'type': 'int',
                        'value': defaults['sensor_size'][0]
                    },
                    {
                        'name': 'width',
                        'type': 'int',
                        'value': defaults['sensor_size'][1]
                    }
                ]
            },
            {
                'name': 'Sensor border',
                'type': 'group',
                'children':
                [
                    {
                        'name': 'y',
                        'type': 'int',
                        'value': defaults['sensor_border'][0]
                    },
                    {
                        'name': 'x',
                        'type': 'int',
                        'value': defaults['sensor_border'][1]
                    },
                ]
            },
            {
                'name': 'Perform sensor search',
                'type': 'bool',
                'value': defaults['perform_sensor_search'],
                'tip': "Search sensor in box"
            },
            {
                'name': 'QR code border',
                'type': 'int',
                'value': defaults['qr_code_border']
            },
            {
                'name': 'Subtract background',
                'type': 'bool',
                'value': defaults['subtract_background'],
                'tip': "Subtract background from signal"
            }
        ]
    },
    {
        'name': 'Advanced parameters',
        'type': 'group',
        'children':
        [
            {
                'name': 'Sensor search area',
                'type': 'group',
                'children':
                [
                    {
                        'name': 'y',
                        'type': 'int',
                        'value': defaults['sensor_search_area'][0]
                    },
                    {
                        'name': 'x',
                        'type': 'int',
                        'value': defaults['sensor_search_area'][1]
                    },
                ]
             },
            {
                'name': 'Sensor threshold factor',
                'type': 'float',
                'value': float(defaults['sensor_thresh_factor']),
                'step': 0.1
            },
            {
                'name': 'Raw auto stretch',
                'type': 'bool',
                'value': defaults['raw_auto_stretch'],
                'tip': "Set to true if raw image intensities should be auto-stretched"
            },
            {
                'name': 'Raw auto wb',
                'type': 'bool',
                'value': defaults['raw_auto_wb'],
                'tip': "Set true if raw image white balance should be set automatically"
            },
            {
                'name': 'Try to correct strip orientation',
                'type': 'bool',
                'value': defaults['strip_try_correct_orientation'],
                'tip': "Will try to find the injection inlet to assess orientation."
            },
            {
                'name': 'Strip orientation correction search rectangles',
                'type': 'group',
                'children':
                [
                    {
                        'name': 'Relative height factor',
                        'type': 'float',
                        'value': defaults['strip_try_correct_orientation_rects'][0],
                        'step': 0.01,
                        'limits': (0, 1)
                    },
                    {
                        'name': 'Relative center cut-off',
                        'type': 'float',
                        'value': defaults['strip_try_correct_orientation_rects'][1],
                        'step': 0.01,
                        'limits': (0, 1)
                    },
                    {
                        'name': 'Relative border cut-off',
                        'type': 'float',
                        'value': defaults['strip_try_correct_orientation_rects'][2],
                        'step': 0.01,
                        'limits': (0, 1)
                    },
                ]
             },
            {
                'name': 'Strip text to search (orientation)',
                'type': 'str',
                'value': defaults['strip_text_to_search'],
                'default': '',
                'tip': "Use prominent text such as COVID"
            },
            {
                'name': 'Strip text is on the right',
                'type': 'bool',
                'value': defaults['strip_text_on_right'],
                'tip': "Set true if test is right of the sensor"
            },
            {
                'name': 'Force FID search',
                'type': 'bool',
                'value': defaults['force_fid_search'],
                'tip': "Set true to force FID search if the QR code with patient data is not present"
            }
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
    'number_of_sensor_bands': 'sensor_bands_number',
    'band_expected_relative_location': 'peak_expected_relative_location',
    'control_band_index': 'control_band_index',
    'force_fid_search': 'force_fid_search',
    'sensor_band_names': 'sensor_band_names',
    'number_of_cores_(max={})'.format(multiprocessing.cpu_count()): 'max_workers',
    'qc': 'qc',
    'verbose': 'verbose'
}
