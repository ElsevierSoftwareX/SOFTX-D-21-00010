import argparse
from pipeline_FH import run_FH
from pathlib import Path
import sys

from pyPOCQuant.lib.settings import load_settings, save_settings
from pyPOCQuant.lib.utils import set_tesseract_exe

__exe__ = "pyPOCQuant"
__version__ = "0.1.0"

if __name__ == '__main__':

    #
    # Parsing input arguments
    #
    parser = argparse.ArgumentParser(description='Automated analysis tool to batch detect and quantify point of care test strips.')

    # Version
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f"{__exe__} {__version__}"
    )

    # Input folder
    parser.add_argument(
        '-f',
        '--folder',
        default='',
        help='folder to be processed'
    )

    # Input folder
    parser.add_argument(
        '-o',
        '--output_folder',
        default='',
        help="output folder where all results are written (if omitted, it defaults to a 'pipeline' subfolder in the input folder)."
    )

    # Create settings file
    parser.add_argument(
        '-c',
        '--create_settings_file',
        default='',
        help='create a settings file with default values'
    )

    # Create settings file
    parser.add_argument(
        '-s',
        '--settings_file',
        default='',
        help='path to the settings file to be used for the analysis'
    )

    # Max number of cores to use
    parser.add_argument(
        '-w',
        '--max_workers',
        default=2,
        help='Max number of cores to use for running the pipeline'
    )

    # Parse the arguments
    args = vars(parser.parse_args())

    # Default parameter values
    DEFAULT_PARAMETERS = {
        "raw_auto_stretch": False,
        "raw_auto_wb": False,
        "strip_text_to_search": "COVID",
        "strip_text_on_right": True,
        "min_sensor_score": 0.85,
        "qr_code_border": 40,
        "sensor_size": (61, 249),
        "sensor_center": (178, 667),
        "subtract_background": True,
        "sensor_border_x": 7,
        "sensor_border_y": 7,
        "perform_sensor_search": True,
        "sensor_thresh_factor": 2,
        "sensor_search_area": (71, 259),
        "peak_expected_relative_location": (0.25, 0.53, 0.79),
        "verbose": True,
        "qc": True
    }

    # Create default settings file?
    if args["create_settings_file"] != "":
        save_settings(
            DEFAULT_PARAMETERS,
            args["create_settings_file"]
        )
        print(f"Default settings written to {args['create_settings_file']}")

        # We can exit now
        sys.exit(0)

    # Set the path to the tesseract executable
    set_tesseract_exe()

    # Get the rest of the arguments

    # Input folder path: contains the images to be processed
    if args["folder"] == "":
        print(f"Please specify an input folder to process.")
        sys.exit(1)

    # Check that the input folder exists
    input_folder_path = Path(args["folder"])
    if not input_folder_path.is_dir():
        print(f"The input folder {input_folder_path} does not exist.")
        sys.exit(1)

    # Result folder
    if args["output_folder"] == "":
        # Defaults to a pipeline subfolder in the input folder
        results_folder_path = Path(input_folder_path / "pipeline")
    else:
        results_folder_path = Path(args["output_folder"])

    # Make sure the output folder exists
    results_folder_path.mkdir(parents=True, exist_ok=True)

    # Settings file
    if args["settings_file"] == "":
        print(f"Please specify a settings file.")
        sys.exit(1)
    settings_file = Path(args["settings_file"])
    if not settings_file.is_file():
        print(f"The settings file {settings_file} does not exist.")
        sys.exit(1)

    # Load the settings
    settings = load_settings(settings_file)

    # Make sure that the settings file is usable
    if settings.keys() != DEFAULT_PARAMETERS.keys():
        print(f"The settings file {settings_file} is not valid.")
        sys.exit(1)

    # Max number of cores
    if args["max_workers"] == "":
        max_workers = 1
    else:
        max_workers = int(args["max_workers"])

    # Inform
    print(f"")
    print(f"Starting analysis with parameters:")
    print(f"                               Input: {input_folder_path}")
    print(f"                              Output: {results_folder_path}")
    print(f"                       Settings file: {Path(settings_file).resolve()}")
    print(f"                   Number of workers: {max_workers}")
    print(f"        RAW auto stretch intensities: {settings['raw_auto_stretch']}")
    print(f"        RAW apply auto white balance: {settings['raw_auto_wb']}")
    print(f"  Strip text to search (orientation): {settings['strip_text_to_search']}")
    print(f"          Strip text is on the right: {settings['strip_text_on_right']}")
    print(f"                           Min score: {settings['min_sensor_score']:.2f}")
    print(f"                      QR code border: {settings['qr_code_border']}")
    print(f"               Perform sensor search: {settings['perform_sensor_search']}")
    print(f"                         Sensor size: {settings['sensor_size']}")
    print(f"                       Sensor center: {settings['sensor_center']}")
    print(f"                  Sensor search area: {settings['sensor_search_area']}")
    print(f"             Sensor threshold factor: {settings['sensor_thresh_factor']}")
    print(f"                     Sensor border x: {settings['sensor_border_x']}")
    print(f"                     Sensor border y: {settings['sensor_border_y']}")
    print(f"    Expected peak relative positions: {settings['peak_expected_relative_location']}")
    print(f"          Subtract signal background: {settings['subtract_background']}")
    print(f"                      Verbose output: {settings['verbose']}")
    print(f"      Create quality-control figures: {settings['qc']}")
    print(f"")

    # Run the pipeline
    run_FH(
        input_folder_path,
        results_folder_path,
        raw_auto_stretch=settings['raw_auto_stretch'],
        raw_auto_wb=settings['raw_auto_wb'],
        strip_text_to_search=settings['strip_text_to_search'],
        strip_text_on_right=settings['strip_text_on_right'],
        min_sensor_score=settings['min_sensor_score'],
        qr_code_border=settings['qr_code_border'],
        perform_sensor_search=settings['perform_sensor_search'],
        sensor_size=settings['sensor_size'],
        sensor_center=settings['sensor_center'],
        sensor_search_area=settings['sensor_search_area'],
        sensor_thresh_factor=settings['sensor_thresh_factor'],
        sensor_border_x=settings['sensor_border_x'],
        sensor_border_y=settings['sensor_border_y'],
        peak_expected_relative_location=settings['peak_expected_relative_location'],
        subtract_background=settings['subtract_background'],
        verbose=settings['verbose'],
        qc=settings['qc'],
        max_workers=max_workers
    )
