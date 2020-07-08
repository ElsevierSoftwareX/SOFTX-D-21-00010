import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import rawpy
from tqdm import tqdm

from pyPOCQuant.lib.analysis import extract_inverted_sensor, analyze_measurement_window, \
    extract_rotated_strip_from_box, get_sensor_contour_fh, use_ocr_to_rotate_strip_if_needed, read_patient_data_by_ocr
from pyPOCQuant.lib.barcode import rotate_if_needed_fh, find_strip_box_from_barcode_data_fh, \
    try_extracting_fid_and_all_barcodes_with_linear_stretch_fh, get_fid_numeric_value_fh, \
    align_box_with_image_border_fh
from pyPOCQuant.lib.consts import Issue
from pyPOCQuant.lib.processing import BGR2Gray
from pyPOCQuant.lib.settings import save_settings
from pyPOCQuant.lib.utils import get_iso_date_from_image, get_exif_details, create_quality_control_images

plt.switch_backend('qt5agg')


def run_pool(files, raw_auto_stretch, raw_auto_wb, input_folder_path,
             results_folder_path, strip_text_to_search, strip_text_on_right,
             min_sensor_score, qr_code_border, perform_sensor_search,
             sensor_size, sensor_center, sensor_search_area, sensor_thresh_factor,
             sensor_border_x, sensor_border_y, peak_expected_relative_location,
             subtract_background, verbose, qc, max_workers=4):
    res = []
    log_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        run_n = partial(run, raw_auto_stretch=raw_auto_stretch, raw_auto_wb=raw_auto_wb,
                        input_folder_path=input_folder_path, results_folder_path=results_folder_path,
                        strip_text_to_search=strip_text_to_search, strip_text_on_right=strip_text_on_right,
                        min_sensor_score=min_sensor_score, qr_code_border=qr_code_border,
                        perform_sensor_search=perform_sensor_search, sensor_size=sensor_size,
                        sensor_center=sensor_center, sensor_search_area=sensor_search_area,
                        sensor_thresh_factor=sensor_thresh_factor,
                        sensor_border_x=sensor_border_x, sensor_border_y=sensor_border_y,
                        peak_expected_relative_location=peak_expected_relative_location,
                        subtract_background=subtract_background, verbose=verbose, qc=qc)
        results = list(tqdm(executor.map(run_n, files), total=len(files)))
    for result in results:
        if result is not None:
            res.append(result[0])
            log_list.append(result[1])
    return res, log_list


def run_FH(
        input_folder_path: Path,
        results_folder_path: Path,
        raw_auto_stretch: bool = False,
        raw_auto_wb: bool = False,
        strip_text_to_search: str = "COVID",
        strip_text_on_right: bool = True,
        min_sensor_score: float = 0.85,
        qr_code_border: int = 30,  # LSR: 30, mobile: 45
        perform_sensor_search: bool = True,
        sensor_size: tuple = (61, 249),
        sensor_center: tuple = (178, 667),
        sensor_search_area: tuple = (71, 259),
        sensor_thresh_factor: float = 2,
        sensor_border_x: int = 7,
        sensor_border_y: int = 7,
        peak_expected_relative_location: tuple = (0.25, 0.53, 0.79),
        subtract_background: bool = True,
        verbose: bool = False,
        qc: bool = False,
        max_workers: int = 2
):
    """Run the whole processing and analysis pipeline.

    Parameters
    ----------

    :param input_folder_path: str
        Folder with the raw images to process.

    :param results_folder_path: str
        Target folder, where all results and quality control figures are written.

    :param raw_auto_stretch: bool
        Whether to automatically correct the white balance of RAW images on load.
        This does not affect JPEG images!

    :param raw_auto_wb: bool
        Whether to automatically stretch image intensities of RAW images on load.
        This does not affect JPEG images!

    :param strip_text_to_search: str
        Text to search on the strip to assess orientation. Set to "" to skip.

    :param strip_text_on_right: bool
        Assuming the strip is oriented horizontally, whether the 'strip_text_to_search' text
        is assumed to be on the right. If 'strip_text_on_right' is True and the text is found on the
        left hand-side of the strip, the strip will be rotated 180 degrees. Ignored if
        strip_text_to_search is "".

    :param min_sensor_score: float
        Minimum segmentation score for the sensor to be considered peak analysis (0.0 <= min_sensor_score <= 1.0).

    :param qr_code_border: int
        Lateral and vertical extension of the (white) border around each QR code.

    :param perform_sensor_search: bool
        If True, the (inverted) sensor is searched within 'sensor_search_area' around the expected 'sensor_center';
        if False, the sensor of size 'sensor_size' is simply extracted from the strip image centered at the
        relative strip position 'sensor_center'.

    :param sensor_size: tuple
        Area of the sensor to be extracted (height, width).

    :param sensor_center:: tuple
        Coordinates of the center of the sensor with respect to the strip image (y, x).

    :param sensor_search_area: tuple
        Search area around the sensor (height, width). Used only if 'skip_sensor_search' is False.

    :param sensor_thresh_factor: int
        Set the number of (robust) standard deviations away from the median band background for a peak
        to be considered valid.

    :param sensor_border_x:
        Lateral sensor border to be ignored in the analysis to avoid border effects.

    :param sensor_border_y:
        Vertical sensor border to be ignored in the analysis to avoid border effects.

    :param peak_expected_relative_location: tuple
        Expected relative peak positions as a function of the width of the sensor (= 1.0)

    :param subtract_background: bool
        If True, estimate and subtract the background of the sensor intensity profile.

    :param verbose: bool
        Toggle verbose output.

    :param qc: bool
        Toggle creation of quality control figures.

    :param max_workers int
        Number of max cores to use for running the pipeline
    """

    # Make sure the results folder exists
    results_folder_path.mkdir(exist_ok=True)

    # Get the list of file names
    filenames = sorted(os.listdir(str(input_folder_path)))

    # Get quantification results
    rows_list, log_list = run_pool(filenames, raw_auto_stretch, raw_auto_wb, input_folder_path,
                                   results_folder_path, strip_text_to_search, strip_text_on_right,
                                   min_sensor_score, qr_code_border, perform_sensor_search,
                                   sensor_size, sensor_center, sensor_search_area, sensor_thresh_factor,
                                   sensor_border_x, sensor_border_y, peak_expected_relative_location,
                                   subtract_background, verbose, qc, max_workers=max_workers)

    # Save data frame
    data = pd.DataFrame(rows_list)
    data.to_csv(str(results_folder_path / "quantification_data.csv"), index=False)
    print(f"Results written to {str(results_folder_path / 'quantification_data.csv')}")

    # Save log
    with open(str(results_folder_path / "log.txt"), 'w+', encoding="utf-8") as f:
        for image_log in log_list:
            f.writelines("%s\n" % item for item in image_log)
        f.close()
    print(f"Logfile written to {str(results_folder_path / 'log.txt')}")

    # Save the settings
    save_settings(
        {
            "raw_auto_stretch": raw_auto_stretch,
            "raw_auto_wb": raw_auto_wb,
            "strip_text_to_search": strip_text_to_search,
            "strip_text_on_right": strip_text_on_right,
            "min_sensor_score": min_sensor_score,
            "qr_code_border": qr_code_border,
            "perform_sensor_search": perform_sensor_search,
            "sensor_size": sensor_size,
            "sensor_center": sensor_center,
            "sensor_search_area": sensor_search_area,
            "sensor_thresh_factor": sensor_thresh_factor,
            "sensor_border_x": sensor_border_x,
            "sensor_border_y": sensor_border_y,
            "peak_expected_relative_location": peak_expected_relative_location,
            "subtract_background": subtract_background,
            "verbose": verbose,
            "qc": qc,
            "file_version": 1
        },
        str(results_folder_path / "settings.txt")
    )
    print(f"Settings written to {str(results_folder_path / 'settings.txt')}")

    # Inform
    print("Pipeline completed.")


def run(
        filename,
        raw_auto_stretch,
        raw_auto_wb,
        input_folder_path: Path,
        results_folder_path: Path,
        strip_text_to_search: str,
        strip_text_on_right: bool,
        min_sensor_score: float = 0.85,
        qr_code_border: int = 30,  # LSR: 30, mobile: 45
        perform_sensor_search: bool = False,
        sensor_size: tuple = (37, 185),
        sensor_center: tuple = (119, 471),
        sensor_search_area: tuple = (50, 195),
        sensor_thresh_factor: float = 2,
        sensor_border_x: int = 7,
        sensor_border_y: int = 7,
        peak_expected_relative_location: tuple = (0.27, 0.55, 0.79),
        subtract_background: bool = True,
        verbose: bool = False,
        qc: bool = False):
    # Initialize the log list
    image_log = []
    image_log.append(f" ")
    image_log.append(f"File = {filename}")

    # Load  the image
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        image = cv2.imread(str(input_folder_path / filename))
    elif filename.lower().endswith(".nef") or \
            filename.lower().endswith(".cr2") or \
            filename.lower().endswith(".arw"):
        with rawpy.imread(str(input_folder_path / filename)) as raw:
            image = raw.postprocess(
                no_auto_bright=not raw_auto_stretch,
                use_auto_wb=raw_auto_wb,
                use_camera_wb=False,
                gamma=(1, 1),
                output_bps=8)

            # Swap channels to BGR to be opencv compatible
            image = image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    else:
        return {}, image_log

    # Initialize results to to add to the dataframe
    results_row = {
        "fid": "",
        "fid_num": -1,
        "filename": filename,
        "extension": os.path.splitext(filename)[1],
        "basename": Path(filename).stem,
        "iso_date": "",
        "iso_time": "",
        "exp_time": "",
        "f_number": "",
        "focal_length_35_mm": "",
        "iso_speed": "",
        "fid_score": 0,
        "manufacturer": "",
        "plate": "",
        "well": "",
        "strip_corr_coeff": 0,
        "sensor_score": 0,
        "ctl": 0,
        "igm": 0,
        "igg": 0,
        "ctl_ratio": 0,
        "igm_ratio": 0,
        "igg_ratio": 0,
        "issue": Issue.NONE.value,
        "transform": ""
    }

    # Inform
    image_log.append(f"Processing {filename}")

    # Extract ISO date from file metadata
    iso_date, iso_time = get_iso_date_from_image(str(input_folder_path / filename))
    results_row["iso_date"] = iso_date
    results_row["iso_time"] = iso_time

    # # Extract additional ISO information
    exp_time, f_number, focal_length_35_mm, iso_speed = get_exif_details(str(input_folder_path / filename))
    results_row["exp_time"] = exp_time
    results_row["f_number"] = f_number
    results_row["focal_length_35_mm"] = focal_length_35_mm
    results_row["iso_speed"] = iso_speed

    # Extract the base name
    basename = Path(filename).stem

    # Store the basename in the results
    results_row["basename"] = basename

    # Find the location of the barcodes
    barcode_data, fid, manufacturer, plate, well, user, best_lb, best_ub, best_score, best_scaling_factor = \
        try_extracting_fid_and_all_barcodes_with_linear_stretch_fh(
            image,
            lower_bound_range=(0, 5, 15, 25, 35),
            upper_bound_range=(100, 98, 95, 92, 89),
            scaling=(0.25, 0.5)
        )

    # Inform
    if verbose:
        image_log.append(
            f"Best percentiles for barcode extraction: ({best_lb}, {best_ub}); "
            f"best scaling factor = {best_scaling_factor}; score = {best_score}/6"
        )

    # Simple check -- if less than three QR codes were found, we have the guarantee that
    # we do not have enough information to extract the strip box.
    # @TODO: Actually check.
    if best_score < 3:
        # Add issue to the results
        results_row["issue"] = Issue.STRIP_BOX_EXTRACTION_FAILED.value
        image_log.append(f"File {filename}: QR/barcode extraction failed. Skipping.")
        row_data = {}
        row_data.update(results_row)
        return row_data, image_log

    # Rotate the image if needed
    image_was_rotated, image, image_log = rotate_if_needed_fh(image, barcode_data, image_log, verbose=verbose)

    # Create quality control images
    if qc:
        create_quality_control_images(
            str(results_folder_path),
            filename.replace('.', '_'),
            {
                "rotated": image
            }
        )

    # If the image was rotated, we need to find the location of the barcodes again.
    # Curiously, re-using the best percentiles we found earlier is *not* guaranteed
    # to succeed. Therefore, we search again.
    # Also, if we managed to extract patient data last time and we fail this time
    # we fall back to the previous values.
    if image_was_rotated:
        barcode_data, new_fid, new_manufacturer, new_plate, new_well, new_user, new_best_lb, \
        new_best_ub, new_best_score, new_best_scaling_factor = \
            try_extracting_fid_and_all_barcodes_with_linear_stretch_fh(
                image,
                lower_bound_range=(0, 5, 15, 25, 35),
                upper_bound_range=(100, 98, 95, 92, 89),
                scaling=(0.25, 0.5)
            )
        fid = new_fid if fid == "" and new_fid != "" else fid
        manufacturer = new_manufacturer if manufacturer == "" and new_manufacturer != "" else manufacturer
        plate = new_plate if plate == "" and new_plate != "" else plate
        well = new_well if well == "" and new_well != "" else well
        user = new_user if user == "" and new_user != "" else user

        # Inform
        if verbose:
            image_log.append(
                f"File {filename}: best percentiles for barcode extraction after "
                f"rotation: ({new_best_lb}, {new_best_ub}); "
                f"best scaling factor = {new_best_scaling_factor}; "
                f"score = {new_best_score}/6"
            )

    # Align the strip box with the image borders (remove potential rotation)
    image, box_rotation_angle = align_box_with_image_border_fh(barcode_data, image)

    # Inform
    if verbose:
        image_log.append(
            f"File {filename}: Strip box image rotated by angle {box_rotation_angle} "
            f"degrees using QR code locations.")

    # Create quality control images
    if qc:
        create_quality_control_images(
            str(results_folder_path),
            filename.replace('.', '_'),
            {
                "aligned_box": image
            }
        )

    # Find location of the strip box from the barcode data
    box, qr_code_extents, qc_image, box_rect = find_strip_box_from_barcode_data_fh(
        image,
        barcode_data,
        qr_code_border=qr_code_border,
        qc=True)

    if box is None or box.shape[0] == 0 or box.shape[1] == 0:
        image_log.append(f"File {filename}: could not extract strip box. Skipping!")

        # Add issue to the results
        results_row["issue"] = Issue.STRIP_BOX_EXTRACTION_FAILED.value
        row_data = {}
        row_data.update(results_row)
        return row_data, image_log

    # Create quality control images
    if qc:
        create_quality_control_images(
            str(results_folder_path),
            filename.replace('.', '_'),
            {
                "box": qc_image
            }
        )
    # If we could not find a valid FID, we try to run OCR in a region
    # a bit larger than the box (in y direction).
    if fid == "":
        box_start_y = box_rect[0] - 600
        if box_start_y < 0:
            box_start_y = 0
        area_for_ocr = image[
                       box_start_y:box_rect[1],
                       box_rect[2]:box_rect[3]
                       ]
        fid, new_manufacturer = read_patient_data_by_ocr(area_for_ocr)
        if manufacturer == "" and new_manufacturer != "":
            manufacturer = new_manufacturer

    results_row["fid"] = fid
    # results_row["fid_score"] = score
    results_row["manufacturer"] = manufacturer
    results_row["plate"] = plate
    results_row["well"] = well
    results_row["user"] = user
    results_row["fid_num"] = get_fid_numeric_value_fh(fid)
    if verbose:
        image_log.append(f"File {filename}: FID = '{fid}'")

    # Do we have a valid FID? If we do not have a valid FID,
    # we can still try and continue with the analysis. Some
    # strips are annotated by hand, and the band result could
    # still be associated to a patient by manually curating the
    # results in the end.
    if fid == "":
        # Add issue to the results
        results_row["issue"] = Issue.FID_EXTRACTION_FAILED.value
        image_log.append(f"File {filename}: could not extract FID. "
                         f"Trying to proceed with the analysis.")

    # Convert box to gray value
    box_gray = BGR2Gray(box)

    # Crop the strip from the box while correcting for possible rotation
    strip_gray, strip = extract_rotated_strip_from_box(box_gray, box)

    # Make a copy of the strip images for analysis
    strip_for_analysis = strip.copy()
    strip_gray_for_analysis = strip_gray.copy()

    # Always save the (aligned) strip image
    create_quality_control_images(
        str(results_folder_path),
        filename.replace('.', '_'),
        {
            "strip_gray_aligned": strip_gray_for_analysis
        }
    )

    # Use tesseract to read text from the strip. If successful,
    # this can be used to figure out the direction in which the
    # strip was placed under the camera.
    strip_gray_for_analysis, strip_for_analysis = \
        use_ocr_to_rotate_strip_if_needed(
            strip_gray_for_analysis,
            strip_for_analysis,
            strip_text_to_search,
            strip_text_on_right
        )

    # Always save the (aligned) strip image
    create_quality_control_images(
        str(results_folder_path),
        filename.replace('.', '_'),
        {
            "strip_gray_aligned_after_ocr": strip_gray_for_analysis
        }
    )

    if perform_sensor_search:

        # Extract the sensor
        sensor, sensor_coords, sensor_score = get_sensor_contour_fh(
            strip_gray_for_analysis,
            sensor_center=sensor_center,
            sensor_size=sensor_size,
            sensor_search_area=sensor_search_area,
            peak_expected_relative_location=peak_expected_relative_location
        )

        if verbose:
            image_log.append(f"File {filename}: sensor coordinates = {sensor_coords}, score = {sensor_score}")

    else:
        # Extract the sensor
        sensor = extract_inverted_sensor(
            strip_gray_for_analysis,
            sensor_center=sensor_center,
            sensor_size=sensor_size
        )
        sensor_score = 1.0

    # Add the sensor score to the results
    results_row["sensor_score"] = sensor_score

    # Always save the sensor image
    create_quality_control_images(
        str(results_folder_path),
        filename.replace('.', '_'),
        {
            "sensor": sensor,
        }
    )

    # Analyse the sensor (it sensor score is >= min_sensor_score)
    if sensor_score >= min_sensor_score:

        # Some (rare) strips have extremely narrow bands. If the normal analysis
        # fails, we try with a narrower peak_width
        peak_widths = [7, 3]
        successful_peak_width = -1
        for curr_peak_width in peak_widths:

            # We have a sensor image and we can proceed with the analysis
            window_results, image_log = analyze_measurement_window(sensor,
                                                                   border_x=sensor_border_x,
                                                                   border_y=sensor_border_y,
                                                                   peak_expected_relative_location=peak_expected_relative_location,
                                                                   subtract_background=subtract_background,
                                                                   thresh_factor=sensor_thresh_factor,
                                                                   peak_width=curr_peak_width,
                                                                   out_qc_folder=results_folder_path,
                                                                   basename=filename.replace('.', '_'),
                                                                   qc=True,
                                                                   verbose=verbose,
                                                                   image_log=image_log)

            if "ctl" in window_results:
                successful_peak_width = curr_peak_width
                break

        # If we have band results, we append them to the dataframe row.

        # The values default to zero, so we do not need to explicitly set them in the
        # corresponding band is missing. If the control band is missing, however, we
        # add flag the issue in the result row.
        if "ctl" in window_results:
            results_row["ctl"] = 1
            results_row["ctl_ratio"] = window_results["ctl"]["normalized_signal"]

            # Inform
            band_type = "normal" if successful_peak_width == peak_widths[0] else "narrow"
            image_log.append(f"File {filename}: the bands were '{band_type}'.")
        else:
            # Inform
            image_log.append(f"File {filename}: the strip does not seem to have a valid control band!")

            # Add issue to the results
            results_row["issue"] = Issue.CONTROL_BAND_MISSING.value

        if "igg" in window_results:
            results_row["igg"] = 1
            results_row["igg_ratio"] = window_results["igg"]["normalized_signal"]

        if "igm" in window_results:
            results_row["igm"] = 1
            results_row["igm_ratio"] = window_results["igm"]["normalized_signal"]

    else:
        # Inform
        image_log.append(f"File {filename}: the sensor did not reach the minimum acceptable "
                         f"score and will be dropped.")

        # The sensor extraction failed. Report the issue
        results_row["issue"] = Issue.SENSOR_EXTRACTION_FAILED.value

    # Save data as dictionary and append to a list of dicts for fast df creation with many rows
    row_data = {}
    row_data.update(results_row)

    # Inform
    if verbose:
        image_log.append(f"✓ File {filename}: successfully processed and added to results table.")

    return row_data, image_log
