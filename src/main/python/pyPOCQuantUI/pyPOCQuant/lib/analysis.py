import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from pyPOCQuant.lib import consts
from pyPOCQuant.lib.barcode import rotate
from pyPOCQuant.lib.processing import BGR2Gray
from pyPOCQuant.lib.utils import get_project_root
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cdist
import sys
import scipy.ndimage as ndimage
from skimage import filters
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from scipy.ndimage import label, find_objects
from sklearn.linear_model import HuberRegressor
from pytesseract import pytesseract
from pytesseract import Output


def analyze_test_area(gray, results_folder_path, basename, low_thres_boundary=190, bar_prominence=5,
                      bar_width=8, qc=True):

    # Invert the image
    gray = invert_image(gray)

    # @todo could be improved if required -> test with pipeline
    ret, thresh = cv2.threshold(gray, low_thres_boundary, 255, 1)

    if qc:
        # @todo add filename based naming
        cv2.imwrite(str(results_folder_path.joinpath(
            basename + '_threshold_control.png')), thresh)

    # Remove some small noise.
    dilate = cv2.dilate(thresh, None)
    erode = cv2.erode(dilate, None)

    # Find contours with cv2.RETR_CCOMP
    if cv2.getVersionMajor() == 4:
        contours, hierarchy = cv2.findContours(
            erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(
            erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Laminar flow element expected to be in the center of the crop (more or less)
    exp_x = gray.shape[1] / 2
    exp_y = gray.shape[0] / 2
    xy1 = np.array([exp_x, exp_y])
    xy1 = np.array([xy1] * 1)

    cnt_x = []
    cnt_y = []
    cnt_w = []
    cnt_h = []
    sensor = gray.copy()
    for i, cnt in enumerate(contours):
        # Check if it is an external contour and its area is more than 100
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_x.append(x)
            cnt_y.append(y)
            cnt_w.append(w)
            cnt_h.append(h)
            # Add bounding box of detected contours
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Add centroid of contours
            cv2.circle(gray, (round(x + w / 2), round(y + h / 2)),
                       3, (255, 0, 0), -1)
    # Add centroid of poc test crop
    cv2.circle(gray, (int(exp_x), int(exp_y)), 3, (0, 0, 255), -1)

    if qc:
        # @todo add filename based naming
        cv2.imwrite(str(results_folder_path.joinpath(
            basename + '_control_img.png')), gray)

    xy2 = np.column_stack(
        (np.add(cnt_x, np.array(cnt_w) / 2), np.add(cnt_y, np.array(cnt_h) / 2)))
    _, min_d_idx = get_min_dist(xy1, xy2)

    sensor = sensor[cnt_y[min_d_idx]:cnt_y[min_d_idx] + cnt_h[min_d_idx],
                    cnt_x[min_d_idx]:cnt_x[min_d_idx] + cnt_w[min_d_idx]]

    if qc:
        cv2.imwrite(str(results_folder_path.joinpath(
            basename + '_sensor.png')), sensor)

    # Compute mean of sensor area column wise to get a better line (peak) profile
    x_val = sensor.mean(axis=0)
    # Detect peaks
    peaks, properties = find_peaks(-x_val,
                                   prominence=bar_prominence, width=bar_width)
    if qc:
        fig, ax = plt.subplots()
        plt.vlines(x=peaks, ymin=x_val[peaks] + properties["prominences"],
                   ymax=x_val[peaks], color="C1")
        plt.hlines(y=-properties["width_heights"], xmin=properties["left_ips"],
                   xmax=properties["right_ips"], color="b")
        plt.plot(x_val)
        plt.plot(peaks, x_val[peaks], "x")
        plt.savefig(str(results_folder_path.joinpath(
            basename + '_peak_detection_control.png')), bbox_inches='tight')
        plt.close(fig)

        for lb in properties["left_ips"]:
            cv2.rectangle(gray, (int(cnt_x[min_d_idx] + lb), int(cnt_y[min_d_idx] - 1)), (int(cnt_x[min_d_idx] + lb + 8),
                                                                                          int(cnt_y[min_d_idx] - 1 + sensor.shape[0])), (255, 255, 0), 1)

    # Note: if not none it is the indices to the peaks needed for alignment of mask
    bar_type = identify_bars(peaks, sensor.shape[1]/2)
    print('Detected bars: {}'.format(bar_type))

    # @todo fix this alignment properly (create_bar_seg_mask)
    mask, mask_bkg = create_bar_seg_mask(sensor.shape, peaks, bar_type)
    mask_m = add_border_margin(mask, margin=4)
    mask_bkg_m = add_border_margin(mask_bkg, margin=4)

    sensor_ctl = sensor.copy()
    _, thresh_m = cv2.threshold(mask_m.astype(
        'uint8'), 0, 255, cv2.THRESH_BINARY)
    _, thresh_m_bkg = cv2.threshold(
        mask_bkg_m.astype('uint8'), 0, 255, cv2.THRESH_BINARY)
    if cv2.getVersionMajor() == 4:
        cnt_m, hierarchy = cv2.findContours(
            thresh_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt_m_bkg, hierarchy = cv2.findContours(
            thresh_m_bkg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, cnt_m, hierarchy = cv2.findContours(
            thresh_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        _, cnt_m_bkg, hierarchy = cv2.findContours(
            thresh_m_bkg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Draw contours:
    cv2.drawContours(sensor_ctl, cnt_m, -1, (0, 255, 0), 1)
    cv2.drawContours(sensor_ctl, cnt_m_bkg, -1, (255, 255, 0), 1)

    if qc:
        fig, ax = plt.subplots()
        plt.subplot(311)
        plt.imshow(sensor)
        plt.subplot(312)
        plt.imshow(mask_m)
        plt.subplot(313)
        plt.imshow(mask_bkg_m)
        plt.savefig(str(results_folder_path.joinpath(
            basename + '_bar_mask_control.png')), bbox_inches='tight')
        plt.close(fig)
        cv2.imwrite(str(results_folder_path.joinpath(
            basename + '_control_sensor_mask.png')), sensor_ctl)

    # Prepare quantified data to be returned
    sensor_inv = invert_image(sensor, bit_depth=8)
    mean_intensity_bkg = np.mean(sensor_inv[mask_bkg_m == 1])
    total_intensity_ctl = sum(sensor_inv[mask_m == 3] - mean_intensity_bkg)
    total_intensity_igm = sum(sensor_inv[mask_m == 1] - mean_intensity_bkg)
    total_intensity_igg = sum(sensor_inv[mask_m == 2] - mean_intensity_bkg)

    # Determine which bars got detected
    # @todo simplify; maybe order of bars does not matter at all. such that the output can simplified to a binary
    if bar_type['ctl'] is not None:
        ctl = 1
    else:
        ctl = 0
    if bar_type['igm'] is not None:
        igm = 1
    else:
        igm = 0
    if bar_type['igg'] is not None:
        igg = 1
    else:
        igg = 0

    data = [[ctl, igm, igg, total_intensity_ctl, total_intensity_igm, total_intensity_igg, mean_intensity_bkg,
             total_intensity_ctl/total_intensity_ctl,  total_intensity_igm/total_intensity_ctl,
             total_intensity_igg/total_intensity_ctl]]

    pd.set_option('display.width', 120)
    pd.set_option('display.max_columns', 10)
    df = pd.DataFrame(data, columns=['CTL', 'IGM', 'IGG', 'CTL_TOTAL', 'IGM_TOTAL', 'IGG_TOTAL', "MEAN_BKG",
                                     'CTL_TOTAL_NORM', 'IGM_TOTAL_NORM', 'IGG_TOTAL_NORM'])
    print(df)
    return df


def add_border_margin(array, margin=2):
    h, w = array.shape
    array_padded = np.zeros((h, w))
    array_padded[margin: h-margin, margin: w -
                 margin] = array[margin: h-margin, margin: w - margin]
    return array_padded


def get_min_dist(xy1, xy2):
    dists = cdist(xy2, xy1, metric='euclidean')
    return np.min(dists), np.argmin(dists)

def identify_bars_alt(peak_positions, profile_length, expected_relative_peak_positions, tolerance=0.1):
    """Assign the peaks to the corresponding bar based on the known relative position in the sensor.

    The bars are in the sequence: IGM, IGG, CTL
    """
    bar_names = ["igm", "igg", "ctl"]
    bars = {}

    relative_peak_positions = (np.array(peak_positions) / profile_length).reshape(-1, 1)
    expected_relative_peak_positions = np.array(expected_relative_peak_positions).reshape(-1, 1)
    dists = cdist(expected_relative_peak_positions, relative_peak_positions, metric='euclidean')

    for i in range(len(dists)):
        dist = dists[i]
        if len(dist) > 0:
            d, indx = np.min(dist), np.argmin(dist)
            if d <= tolerance:
                bars[bar_names[i]] = indx

    return bars

def identify_bars(peaks, center_x, tolerance=15):
    """ Aims to identify the type ot the bars (control := 0, igg:=2, igm:=1) \
        and its location in respect to the laminar flow bounding box as not \
        all bars might be present. Tolerance is in units pixel"""

    # define expected location of the bars in respect to the bounding box
    exp_igm_x = center_x / 2
    exp_control_x = center_x * 1.5

    dists_to_center = [abs(i-center_x) for i in peaks]
    dists_to_control = [abs(i-exp_control_x) for i in peaks]
    dists_to_igm = [abs(i-exp_igm_x) for i in peaks]

    # Determine presence, type and indices of bars with its peak location
    ret = {}

    if min(dists_to_control) <= tolerance:
        idx = dists_to_control.index(min(dists_to_control))
        ret['ctl'] = idx
    else:
        ret['ctl'] = None

    if min(dists_to_igm) <= tolerance:
        idx = dists_to_igm.index(min(dists_to_igm))
        ret['igm'] = idx
    else:
        ret['igm'] = None

    if min(dists_to_center) <= tolerance:
        idx = dists_to_center.index(min(dists_to_center))
        ret['igg'] = idx
    else:
        ret['igg'] = None

    return ret


def create_bar_seg_mask(laminar_flow_shape, peaks, bar_type, bar_width=10, inter_bar_dist=40):

    # @todo This needs major revision!
    # What it should be doing is (i) detect if there is at least a bar, if so create mask aligned by the bar or if
    # multiple bars align mask such that it minimizes its distance to all bars. (ii) if there is no bar center it around
    # sensor centroid.
    # Important. Not use peaks! They can be shifted. Use full half width of peak center and align with this one
    # -> new method with peaks and full half widths to 're-center'

    if bar_type['ctl'] is not None:
        c_idx = list(range(peaks[bar_type['ctl']]-(round(bar_width/2)-1),
                           peaks[bar_type['ctl']]+round(bar_width/2)))
        m_idx = [i - 2*inter_bar_dist for i in c_idx]
        g_idx = [i - inter_bar_dist for i in c_idx]

        # indices = c_idx + g_idx + m_idx
        bkg_idx = list(range(max(m_idx)+1, min(g_idx))) + \
            list(range(max(g_idx)+1, min(c_idx)))

        mask_row = np.zeros(laminar_flow_shape[1], dtype=int)
        mask_row[c_idx] = 3
        mask_row[m_idx] = 1
        mask_row[g_idx] = 2

        mask_bkg_row = np.zeros(laminar_flow_shape[1], dtype=int)
        mask_bkg_row[bkg_idx] = 1

        mask_out = np.array([mask_row]*laminar_flow_shape[0])
        mask_bkg_out = np.array([mask_bkg_row] * laminar_flow_shape[0])

    elif bar_type['igm']:
        g_idx = list(range(peaks[bar_type['igm']]-round(bar_width/2),
                           peaks[bar_type['igm']]+round(bar_width/2)))
        m_idx = [i + inter_bar_dist for i in g_idx]
        c_idx = [i + 2*inter_bar_dist for i in g_idx]

        indices = c_idx + g_idx + m_idx

        mask_row = np.zeros(laminar_flow_shape[1], dtype=int)

        mask_row[indices] = 1

        mask_out = np.array([mask_row]*laminar_flow_shape[0])

    elif bar_type['igg']:
        g_idx = list(range(peaks[bar_type['igg']]-round(bar_width/2),
                           peaks[bar_type['igg']]+round(bar_width/2)))
        m_idx = [i + inter_bar_dist for i in g_idx]
        c_idx = [i - inter_bar_dist for i in g_idx]

        indices = c_idx + g_idx + m_idx

        mask_row = np.zeros(laminar_flow_shape[1], dtype=int)

        mask_row[indices] = 2

        mask_out = np.array([mask_row]*laminar_flow_shape[0])
    else:
        mask_out = None
        mask_bkg_out = None

    return mask_out, mask_bkg_out


def invert_image(image, bit_depth=8):
    image_inv = (2**bit_depth - image)
    return image_inv.astype('uint8')


def local_minima(array, min_distance=1):
    """Find all local minima of the array, separated by at least min_distance."""
    max_points = array == ndimage.maximum_filter(
        array, 1 + 2 * min_distance, mode='constant', cval=array.max() + 1)
    return np.array([indices[max_points] for indices in np.indices(array.shape)])


def find_peak_bounds(profile, border, peak_index, image_log, verbose=False):

    profile = np.asarray(profile)

    # Do not go into the border!
    lowest_bound = border
    highest_bound = len(profile) - border

    # Peak intensity
    peak_intensity = profile[peak_index]

    # Not move away from the peak in both directions until the intensity in under 'relative_intensity'
    current_lower_background = peak_intensity
    n = 0
    current_lower_bound = peak_index - 1
    for index in range(peak_index - 1, lowest_bound, -1):
        if profile[index] <= current_lower_background:
            current_lower_background = profile[index]
            current_lower_bound = index
        else:
            # If any of the next indices (up to a max of three) is lower than
            # 'current_background', we still allow them to be picked; to avoid
            # picking a noisy bump instead.
            if n > 1:
                break
            n += 1

    # On the other side
    current_upper_background = peak_intensity
    n = 0
    current_upper_bound = peak_index + 1
    for index in range(peak_index + 1, highest_bound):
        if profile[index] <= current_upper_background:
            current_upper_background = profile[index]
            current_upper_bound = index
        else:
            # If any of the next indices (up to a max of three) is lower than
            # 'current_background', we still allow them to be picked; to avoid
            # picking a noisy bump instead.
            if n > 1:
                break
            n += 1

    # Now do some quality check on the extracted bounds
    d_lower = peak_index - current_lower_bound
    d_upper = current_upper_bound - peak_index
    background = current_lower_background if current_lower_background < current_upper_background else current_upper_background
    i_lower = (current_lower_background - background) / (peak_intensity - background)
    i_upper = (current_upper_background - background) / (peak_intensity - background)
    band_skewness = d_upper / d_lower

    if verbose:
        # print(f"Peak {peak_index} has lower bound {current_lower_bound} (d = {d_lower}) "
        #       f"with relative intensity {i_lower:.2f} and "
        #       f"upper bound {current_upper_bound} (d = {d_upper}) with relative intensity {i_upper:.2f}. "
        #       f"Band width is {current_upper_bound - current_lower_bound + 1}. Band skewness is {band_skewness:.2f}")
        image_log.append(f"Peak {peak_index} has lower bound {current_lower_bound} (d = {d_lower}) "
              f"with relative intensity {i_lower:.2f} and "
              f"upper bound {current_upper_bound} (d = {d_upper}) with relative intensity {i_upper:.2f}. "
              f"Band width is {current_upper_bound - current_lower_bound + 1}. Band skewness is {band_skewness:.2f}")

    return current_lower_bound, current_upper_bound, image_log


def fit_and_subtract_background(profile, border, subtract_offset=10):
    """Use a robust linear estimator to estimate the background of the profile and subtract it."""

    # Prepare data
    y = profile[border:-border].squeeze()
    x = np.arange(y.size).reshape(-1, 1)

    # Instantiate the model
    model = HuberRegressor(fit_intercept=True)

    # Fit
    model.fit(x, y)

    # Predict
    y_hat = model.predict(x)

    # Subtract the background
    subtr = y - (y_hat - subtract_offset)

    # Insert in the original profile
    profile[border:-border] = subtr

    # Also return the predicted background and the predicted background with offset
    background = 0 * profile.copy()
    background[border:-border] = y_hat

    background_offset = 0 * profile.copy()
    background_offset[border:-border] = y_hat - subtract_offset

    return profile, background, background_offset


def estimate_threshold_for_significant_peaks(profile, border_x, thresh_factor):

    # First find all local minima (add back the border offset)
    loc_min_indices = border_x + local_minima(profile[border_x: len(profile) - border_x])

    # Calculate a statistical threshold for peaks using the local min intensities.
    min_values = profile[loc_min_indices]
    md = np.median(min_values)
    ma = np.median(np.abs(min_values - md))
    peak_threshold = md + thresh_factor * ma

    # Now remove the outliers from the local minima
    new_min_values = min_values[min_values < peak_threshold]

    # Calculate the new threshold
    md = np.median(new_min_values)
    ma = np.median(np.abs(new_min_values - md))
    peak_threshold = md + thresh_factor * ma

    # Keep only the local minima that passed both tests
    loc_min_indices = loc_min_indices[profile[loc_min_indices] < peak_threshold]
    if np.any(profile[loc_min_indices] >= peak_threshold):
        raise Exception("Logical flaw in estimate_threshold_for_significant_peaks().")

    return peak_threshold, loc_min_indices, md


def analyze_measurement_window(
    window,
    border_x=10,
    border_y=5,
    thresh_factor=3,
    peak_width=7,
    peak_expected_relative_location=(0.27, 0.55, 0.79),
    subtract_background=False,
    qc=False,
    verbose=False,
    out_qc_folder='',
    basename='',
    image_log=[]):
    """Quantify the band signal across the sensor.

    Notice: the expected relative peak positions for the original strips were: [0.30, 0.52, 0.74]
    """

    # Initialize profile
    profile = np.zeros(window.shape[1])

    # Process all columns
    for c in range(border_x, window.shape[1] - border_x):

        # Extract the column (without border)
        column = window[border_y: window.shape[0] - border_y, c]

        # Calculate and store the mean value
        profile[c] = np.mean(column)

    # Subtract the background
    if subtract_background:

        if qc:
            original_profile = profile.copy()

        # Estimate and subtract the background
        profile, background, background_offset = fit_and_subtract_background(profile, border_x, subtract_offset=20)

        # Quality control plots
        if qc:

            fig, ax = plt.subplots()

            # Plot profile and estimated background
            ax.plot(
                np.arange(border_x, len(original_profile) - border_x),
                original_profile[border_x: len(original_profile) - border_x],
                'k-',
                markersize=6)
            ax.plot(
                np.arange(border_x, len(background) - border_x),
                background[border_x: len(background) - border_x],
                'k--',
                markersize=6)
            ax.plot(
                np.arange(border_x, len(background_offset) - border_x),
                background_offset[border_x: len(background_offset) - border_x],
                'r-',
                markersize=6)

            # Save to output folder
            filename = str(Path(out_qc_folder) / (basename + "_peak_background_estimation.png"))
            fig.savefig(filename)
            plt.close(fig)


    # Estimate a threshold (on the noisy data) to distinguish noisy candidate peaks from likely correct ones
    peak_threshold, loc_min_indices, md = estimate_threshold_for_significant_peaks(profile, border_x, thresh_factor)

    # Low-pass filter the profile
    profile[border_x:-border_x] = gaussian_filter1d(profile[border_x:-border_x], 1)

    # Find the peaks (add back the border offset)
    peaks = find_peaks(profile[border_x: len(profile) - border_x], width=peak_width)[0] + border_x

    # Integrate the band signals
    valid_peaks = []
    valid_lower_bounds = []
    valid_upper_bounds = []
    for index, c_peak in enumerate(peaks):
        # If the absolute peak value is under the peak_threshold,
        # drop it and move on to the next candidate.
        if profile[c_peak] < peak_threshold:
            continue

        # Find the peak bounds
        lower_bound, upper_bound, image_log = find_peak_bounds(profile, border_x, c_peak, image_log, verbose)

        # Store the information
        valid_peaks.append(c_peak)
        valid_lower_bounds.append(lower_bound)
        valid_upper_bounds.append(upper_bound)

    # Check that the peaks do not overlap
    for i in range(len(valid_lower_bounds) - 1):
        next_lower_bound = valid_lower_bounds[i + 1]
        current_upper_bound = valid_upper_bounds[i]
        if current_upper_bound >= next_lower_bound:
            split = (current_upper_bound + next_lower_bound) // 2
            valid_lower_bounds[i + 1] = split
            valid_upper_bounds[i] = split - 1

    # Finally integrate the signal
    band_signals = []
    for c_peak, lower_bound, upper_bound in zip(valid_peaks, valid_lower_bounds, valid_upper_bounds):
        # Integrate the signal
        dy = (profile[upper_bound] - profile[lower_bound]) / (upper_bound - lower_bound + 1)
        tot_intensity = 0.0
        i = 0
        for c in range(lower_bound, upper_bound + 1):
            tot_intensity += profile[c] - (profile[lower_bound] + float(i) * dy)
            i += 1

        band_signals.append(tot_intensity)

    # Now assign the peaks to the physical band location
    bars = identify_bars_alt(valid_peaks, len(profile), peak_expected_relative_location)

    # Merge quantification and bars dictionary
    merged_results = {}
    for bar in bars:
        indx = int(bars[bar])
        if indx is not None:
            current = {
                "band": bar,
                "peak_pos": valid_peaks[indx],
                "signal": band_signals[indx],
                "normalized_signal": 0.0,
                "peak_index": indx
                }
            merged_results[bar] = current

    # Update the valid_peaks, valid_lower_bounds, and upper_lower_bounds lists
    valid_peaks_original = valid_peaks.copy()
    valid_lower_bounds_original = valid_lower_bounds.copy()
    valid_upper_bounds_original = valid_upper_bounds.copy()
    valid_peaks = []
    valid_lower_bounds = []
    valid_upper_bounds = []
    for bar in bars:
        indx = bars[bar]
        if indx is not None:
            valid_peaks.append(valid_peaks_original[indx])
            valid_lower_bounds.append(valid_lower_bounds_original[indx])
            valid_upper_bounds.append(valid_upper_bounds_original[indx])

    # Now normalize the signals against the control
    if "ctl" in merged_results:
        ctl_signal = merged_results["ctl"]["signal"]
        merged_results["ctl"]["normalized_signal"] = 1.0

        if "igg" in merged_results:
            merged_results["igg"]["normalized_signal"] = merged_results["igg"]["signal"] / ctl_signal

        if "igm" in merged_results:
            merged_results["igm"]["normalized_signal"] = merged_results["igm"]["signal"] / ctl_signal

    # Quality control plots
    if qc:

        fig, ax = plt.subplots()

        # Plot profile
        ax.plot(
            np.arange(border_x, len(profile) - border_x), profile[border_x: len(profile) - border_x],
            'k-', markersize=6)

        ax.set_xlim([0, len(profile)])
        ax.set_ylim([
            np.min(profile[loc_min_indices]) * 0.9,
            np.max(profile) * 1.1])

        # Plot minima
        for min in loc_min_indices:
            ax.plot(min, profile[min], 'g.')

        # Plot peaks and local bounds
        for peak in valid_peaks:
            ax.plot(peak, profile[peak], 'rs', markersize=4)
        for lower_bound, upper_bound in zip(valid_lower_bounds, valid_upper_bounds):
            ax.plot([lower_bound, upper_bound], [profile[lower_bound], profile[upper_bound]], 'o-', linewidth=2)

        # Plot the estimated background
        ax.plot([0, len(profile)], [peak_threshold, peak_threshold], 'r--')

        # Plot the peak threshold
        ax.plot([0, len(profile)], [md, md], 'g--')

        # Save to output folder
        filename = str(Path(out_qc_folder) / (basename + "_peak_analysis.png"))
        fig.savefig(filename)
        plt.close(fig)

        # Draw the band on the original image
        fig, ax = plt.subplots()
        ax.imshow(window, cmap='gray')
        for lower_bound, upper_bound in zip(valid_lower_bounds, valid_upper_bounds):
            ax.plot([lower_bound, upper_bound, upper_bound, lower_bound, lower_bound],
                     [border_y, border_y, window.shape[0] - border_y, window.shape[0] - border_y, border_y],
                     '-', linewidth=2)

        # Save to output folder
        filename = str(Path(out_qc_folder) / (basename + "_peak_overlays.png"))
        fig.savefig(filename)
        plt.close(fig)

    return merged_results, image_log


def threshold_image_original(gray, output_shape=(40, 175)):

    # Define horizontal | and vertical __ kernel.
    kernel_horiz = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]])

    kernel_vert = kernel_horiz.T

    # Calc the gradient of the windowed image
    gray_grad, _ = calculate_gradient_magnitude(gray, suppress_double_edge=False)

    # Blur image and compute the difference of gaussian
    blur1 = cv2.GaussianBlur(gray_grad, (5, 5), 1, 1)
    blur2 = cv2.GaussianBlur(gray_grad, (5, 5), 2.0, 2.0)
    DoG_edge = blur1 - blur2
    DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_horiz)
    DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_vert)
    DoG_edge[DoG_edge <= 0] = 0

    # Calc mean profiles along both image axes
    dh = np.mean(DoG_edge, 1)
    dw = np.mean(DoG_edge, 0)

    # Calc mean, mad and median for both mean profiles
    ds_w = [np.mean(dw), np.mean(np.absolute(dw - np.mean(dw))), np.median(dw)]
    ds_h = [np.mean(dh), np.mean(np.absolute(dh - np.mean(dh))), np.median(dh)]

    # Test range with mean + k * mad to find optimal aspect ratio
    dwc = dw.copy()
    dhc = dh.copy()
    aspect_ratio = []
    widths = []
    heights = []
    peak_widths = []
    peak_heights = []
    for k in np.arange(0, 5, 0.1):

        # Set values on copy of mean profiles to 0 if below mean + k * mad
        dwc[dw < (ds_w[0] + k * ds_w[1])] = 0
        dhc[dh < (ds_h[0] + k * ds_h[1])] = 0

        # Find peaks with a prominence larger than mad
        p_width, _ = find_peaks(dwc, prominence=ds_w[1])
        if len(p_width) == 0:
            p_width = [0, 0]
        p_height, _ = find_peaks(dhc, prominence=ds_h[1])
        if len(p_height) == 0:
            p_height = [0, 0]

        width = max(p_width) - min(p_width)
        height = max(p_height) - min(p_height)
        peak_widths.append(p_width)
        peak_heights.append(p_height)
        widths.append(width)
        heights.append(height)
        if height != 0:
            aspect_ratio.append(width/height)
        else:
            aspect_ratio.append(0)

    # Compute the distance of the detected to the expected aspect ratio and get the index of the minimal distance
    exp_ratio = output_shape[1] / output_shape[0]
    ratio_dist = [abs(x - exp_ratio) for x in aspect_ratio]
    ratio_dist_idx = ratio_dist.index(min(ratio_dist))

    # @todo add aspect ratio check and return distance to expected value for aspect, width and height
    width = widths[ratio_dist_idx]
    height = heights[ratio_dist_idx]
    peak_width = peak_widths[ratio_dist_idx]
    peak_height = peak_heights[ratio_dist_idx]

    # Calculate the interection over union score
    score = calc_intersection_over_union_score(
        max(peak_width) - min(peak_width),
        max(peak_height) - min(peak_height),
        output_shape[1],
        output_shape[0]
    )

    # Compute centroid coordinates and top left
    x_c = round(((max(peak_width) - min(peak_width)) / 2) + min(peak_width))
    y_c = round(((max(peak_height) - min(peak_height)) / 2) + min(peak_height))
    xy1 = (int(x_c - (width / 2)), int(y_c - (height / 2)))
    xy2 = (int(x_c - (width / 2) + width), int(y_c - (height / 2) + height))

    out = xy1[0], x_c, xy1[1], y_c, width, height
    mask = np.zeros(DoG_edge.shape, np.uint8)
    # Add 8 bit max intensity
    mask[xy1[1]:xy1[1] + height, xy1[0]:xy1[0] + width] = 255
    thresh = mask

    # Remove some small noise.
    dilate = cv2.dilate(thresh, None)
    erode = cv2.erode(dilate, None)
    return erode, thresh, score


def threshold_image(gray, method=1, output_shape=(40, 175), exp_center=None):

    # Define horizontal | and vertical __ kernel.
    kernel_horiz = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]])

    kernel_vert = kernel_horiz.T
    kernel_cross = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]], np.uint8)
    kernel_square = np.ones((5, 5), np.uint8)

    if method == 1:

        # https://en.wikipedia.org/wiki/Difference_of_Gaussians
        # Kernel size 5
        # sigma 1 and sigma 2
        print('Threshold using Difference of Gaussian')
        print('Kernel size (5,5), sigma1_xy = 1, sigma2_xy = 2')

        blur1 = cv2.GaussianBlur(gray, (5, 5), 1, 1)
        blur2 = cv2.GaussianBlur(gray, (5, 5), 2.0, 2.0)
        DoG_edge = blur1 - blur2
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_horiz)
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_vert)


        _, thresh = cv2.threshold(DoG_edge, 0, 255, cv2.THRESH_OTSU)

        score = None

    elif method == 2:

        print('Threshold using OTSU')
        print('Kernel size (5,5) for bluring')
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        smooth = cv2.addWeighted(blur, 1.5, gray, -0.5, 0)
        ret3, thresh = cv2.threshold(smooth, 0, 255, cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)

        score = None

    elif method == 3:
        blur1 = cv2.GaussianBlur(gray, (5, 5), 1, 1)
        blur2 = cv2.GaussianBlur(gray, (5, 5), 2.0, 2.0)
        DoG_edge = blur1 - blur2
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_horiz)
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_vert)

        _, thresh_o = cv2.threshold(DoG_edge, 0, 255, cv2.THRESH_OTSU)

        # Apply cross filter to strengthen vertical lines
        thresh_cross_filt = cv2.filter2D(thresh_o, -1, kernel_cross)
        # Morph closing
        closing = cv2.morphologyEx(thresh_cross_filt, cv2.MORPH_CLOSE, kernel_square)
        dilation = cv2.dilate(closing, kernel_square, iterations=5)
        # Fill holes in mask
        filled = fill_mask(dilation.copy())
        # Errod back to orig shape
        thresh = cv2.erode(filled, kernel_square, iterations=5)

        score = None

    elif method == 4:
        # Calc the gradient of the windowd image
        gray_grad, _ = calculate_gradient_magnitude(gray, suppress_double_edge=False)

        # Blur image and compute the difference of gaussian
        blur1 = cv2.GaussianBlur(gray_grad, (5, 5), 1, 1)
        blur2 = cv2.GaussianBlur(gray_grad, (5, 5), 2.0, 2.0)
        DoG_edge = blur1 - blur2
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_horiz)
        DoG_edge = cv2.filter2D(DoG_edge, -1, kernel_vert)
        DoG_edge[DoG_edge <= 0] = 0

        # Calc mean profiles along both image axes
        dh = np.mean(DoG_edge, 1)
        dw = np.mean(DoG_edge, 0)

        # Calc mean, mad and media for both mean profiles
        ds_w = [np.mean(dw), np.mean(np.absolute(dw - np.mean(dw))), np.median(dw)]
        ds_h = [np.mean(dh), np.mean(np.absolute(dh - np.mean(dh))), np.median(dh)]
        # plt.plot(dw)
        # plt.show()

        # Test range with mean + k * mad to find optimal aspect ratio
        dwc = dw.copy()
        dhc = dh.copy()
        aspect_ratio = []
        widths = []
        heights = []
        peak_widths = []
        peak_heights = []
        exp_lower_edge_w = exp_center[0] - output_shape[1]/2
        exp_upper_edge_w = exp_center[0] + output_shape[1]/2
        exp_lower_edge_h = exp_center[1] - output_shape[0]/2
        exp_upper_edge_h = exp_center[1] + output_shape[0]/2
        for k in np.arange(0, 5, 0.1):

            # Set values on copy of mean profiles to 0 if below mean + k * mad
            dwc[dw < ((ds_w[0] - 0.15 * ds_w[0]) + k * ds_w[1])] = 0
            dhc[dh < ((ds_h[0] - 0.15 * ds_h[0]) + k * ds_h[1])] = 0

            # Find peaks with a prominence larger than mad
            p_width, _ = find_peaks(dwc, prominence=ds_w[1])
            if len(p_width) == 0:
                p_width = [0, 0]
            p_height, _ = find_peaks(dhc, prominence=ds_h[1])
            if len(p_height) == 0:
                p_height = [0, 0]

            # Width edge distances
            d_w_lower_edge = [abs(exp_lower_edge_w - p) for p in p_width]
            d_w_upper_edge = [abs(exp_upper_edge_w - p) for p in p_width]

            width, _, _ = get_pairwise_minimal_dist(p_width, output_shape[1], d_w_lower_edge, d_w_upper_edge)

            # Height edge distances
            d_h_lower_edge = [abs(exp_lower_edge_h - p) for p in p_height]
            d_h_upper_edge = [abs(exp_upper_edge_h - p) for p in p_height]

            height, _, _ = get_pairwise_minimal_dist(p_height, output_shape[0], d_h_lower_edge, d_h_upper_edge)

            # width = max(p_width) - min(p_width)
            # height = max(p_height) - min(p_height)
            peak_widths.append(p_width)
            peak_heights.append(p_height)
            widths.append(width)
            heights.append(height)
            if height != 0:
                aspect_ratio.append(width/height)
            else:
                aspect_ratio.append(0)

        # Compute the distance of the detected to the expected aspect ratio and get the index of the minimal distance
        exp_ratio = output_shape[1] / output_shape[0]
        ratio_dist = [abs(x - exp_ratio) for x in aspect_ratio]
        ratio_dist_idx = ratio_dist.index(min(ratio_dist))

        # @todo add aspect ratio check and return distance to expected value for aspect, width and height
        width = widths[ratio_dist_idx]
        height = heights[ratio_dist_idx]
        peak_width = peak_widths[ratio_dist_idx]
        peak_height = peak_heights[ratio_dist_idx]

        # Calculate the interection over union score
        score = calc_intersection_over_union_score(
            max(peak_width) - min(peak_width),
            max(peak_height) - min(peak_height),
            output_shape[1],
            output_shape[0]
        )

        # Compute centroid coordinates and top left
        x_c = round(((max(peak_width) - min(peak_width)) / 2) + min(peak_width))
        y_c = round(((max(peak_height) - min(peak_height)) / 2) + min(peak_height))
        xy1 = (int(x_c - (width / 2)), int(y_c - (height / 2)))
        xy2 = (int(x_c - (width / 2) + width), int(y_c - (height / 2) + height))

        out = xy1[0], x_c, xy1[1], y_c, width, height
        mask = np.zeros(DoG_edge.shape, np.uint8)
        # Add 8 bit max intensity
        print('Applying width {}, height {}'.format(width, height))
        mask[xy1[1]:xy1[1] + height, xy1[0]:xy1[0] + width] = 255
        thresh = mask

    else:
        print('Unknown threshold method')
        return np.zeros(gray.shape, np.uint8), np.zeros(gray.shape, np.uint8)

    # Remove some small noise. 
    dilate = cv2.dilate(thresh, None)
    erode = cv2.erode(dilate, None)
    return erode, thresh, score


def extract_inverted_sensor(gray, sensor_center=(119, 471), sensor_size=(40, 190)):
    """Returns the sensor area at the requested position without searching."""
    x0 = sensor_center[1] - sensor_size[1] // 2
    x = x0 + sensor_size[1]
    y0 = sensor_center[0] - sensor_size[0] // 2
    y = y0 + sensor_size[0]
    return invert_image(gray[y0:y, x0:x])


def get_pairwise_minimal_dist(side_length, exp_side_length, dist_lower_edge, dist_upper_edge):
    I, J = np.triu_indices(len(side_length), 1)

    side_length_dist = []
    edge_dist_sum = []
    dist_total = []
    if len(I) == 0:
        return side_length[0], 0, 0
    else:
        for i in range(0, len(I)):
            side_length_dist.append(abs(exp_side_length - abs(side_length[I[i]] - side_length[J[i]])))
            edge_dist_sum.append(dist_lower_edge[I[i]] + dist_upper_edge[J[i]])
            dist_total.append(side_length_dist[i] + edge_dist_sum[i])
        idx = dist_total.index(min(dist_total))
        optimal_side_length = abs(side_length[J[idx]] - side_length[I[idx]])
        return optimal_side_length, I[idx], J[idx]


def calc_intersection_over_union_score(peak_width, peak_height, width, height):
    """Calculate the intersection over union as a score for the similariy of the
    found rectangle against the expected one. Maximum score is 1.0, minimum is 0.0."""
    target = np.full((max(peak_height, height), max(peak_width, width)), False, dtype=bool)
    calc_rect = target.copy()
    ground_truth_rect = target.copy()
    calc_rect[0: peak_height, 0: peak_width] = True
    ground_truth_rect[0: height, 0: width] = True
    return (np.sum(calc_rect & ground_truth_rect) / np.sum(calc_rect | ground_truth_rect))


def find_central_contour(image, image_masked):

    # Find contours with cv2.RETR_CCOMP
    if cv2.getVersionMajor() == 4:
        contours, hierarchy = cv2.findContours(
            image_masked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv2.findContours(
            image_masked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Sensor expected to be more or less in the center of the image.
    exp_x = image.shape[1] / 2
    exp_y = image.shape[0] / 2
    xy1 = np.array([exp_x, exp_y])
    xy1 = np.array([xy1] * 1)

    cnt_x = []
    cnt_y = []
    cnt_w = []
    cnt_h = []
    img2 = image.copy()
    for i, cnt in enumerate(contours):
        # Check if it is an external contour and its area is more than 100
        if hierarchy[0, i, 3] == -1 and cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cnt_x.append(x)
            cnt_y.append(y)
            cnt_w.append(w)
            cnt_h.append(h)
            # Add bounding box of detected contours
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Add centroid of contours
            cv2.circle(img2, (round(x + w / 2), round(y + h / 2)),
                       3, (255, 0, 0), -1)

    # Add centroid of poc test crop
    cv2.circle(img2, (int(exp_x), int(exp_y)), 3, (0, 0, 255), -1)

    # Did we find any valid contours?
    if len(cnt_x) == 0:
        return None, img2

    # Find closest contour to image center and print its properties
    xy2 = np.column_stack(
        (np.add(cnt_x, np.array(cnt_w) / 2), np.add(cnt_y, np.array(cnt_h) / 2)))
    _, min_d_idx = get_min_dist(xy1, xy2)

    # Return the contour of the sensor area
    contour = [cnt_x[min_d_idx], cnt_y[min_d_idx], cnt_w[min_d_idx], cnt_h[min_d_idx]]

    return contour, img2


def fill_mask(sensor_area_masked):

    filled_bkg = sensor_area_masked.copy()
    cv2.floodFill(filled_bkg, None, (0, 0), 255)

    height, width = sensor_area_masked.shape[:2]
    for x in range(height):
        for y in range(width):
            if filled_bkg[x, y] == 0:
                cv2.floodFill(sensor_area_masked, None, (y, x), 255)

    # Erode pixel noise
    sensor_area_masked_out = cv2.erode(sensor_area_masked, None,  iterations=2)
    return sensor_area_masked_out


def get_sensor_contour_fh(
        strip_gray,
        sensor_center,
        sensor_size,
        sensor_search_area,
        peak_expected_relative_location,
        min_control_bar_width=7
    ):
    """Extract the sensor area from the gray strip image."""

    # Invert the image so that the bands are white
    bit_depth = 8 if strip_gray.dtype == np.uint8 else 16
    strip_gray_inverted = invert_image(strip_gray.copy(), bit_depth)

    # Extract the search area
    dy = sensor_search_area[0] // 2
    dx = sensor_search_area[1] // 2
    search_area = strip_gray_inverted[
                  sensor_center[0] - dy: sensor_center[0] + dy + 1,
                  sensor_center[1] - dx: sensor_center[1] + dx + 1,
                  ]

    # Segment using Li
    optimal_threshold = filters.threshold_li(search_area)
    search_area_bw = search_area > optimal_threshold

    # Now find the control band starting from the right
    profile = np.zeros(search_area_bw.shape[1])
    for x in range(search_area_bw.shape[1] - 1, -1, -1):
        profile[x] = np.sum(search_area_bw[:, x])

    # Find the bars
    peaks, properties = find_peaks(profile, prominence=5, width=8)

    # Keep bars that are wide enough as candidates
    candidate_locations = []
    candidate_relative_locations = []
    candidate_widths = []
    for i in range(len(peaks)):
        width = properties["widths"][i]
        if width > min_control_bar_width:
            candidate_locations.append(peaks[i])
            candidate_relative_locations.append(float(peaks[i]) / sensor_search_area[1])
            candidate_widths.append(width)

    # Keep bar that is closest to the expected position
    # (even considering the mistake given by the largest
    # search area)
    accepted_loc = -1
    accepted_width = -1
    if len(candidate_locations) > 0:
        indx = np.argmin(
            cdist(
                np.array(peak_expected_relative_location[2]).reshape(-1, 1),
                np.array(candidate_relative_locations).reshape(-1, 1),
            )
        )
        accepted_loc = candidate_locations[indx]
        accepted_width = candidate_widths[indx]

    # If nothing was found, just return the center of the
    # search area (with the expected size)
    if accepted_loc == -1:
        # Return the center of the search_area
        y0 = sensor_center[0] - sensor_size[0] // 2
        y = y0 + sensor_size[0]
        x0 = sensor_center[1] - sensor_size[1] // 2
        x = x0 + sensor_size[1]

        # @ToDo Come up with a proper score
        sensor_score = 0.0

        return strip_gray_inverted[y0: y, x0: x], [y0, y, x0, x], sensor_score

    # Now use the expected peak location to re-center the sensor in x
    corrected_pos = accepted_loc - (sensor_search_area[1] - sensor_size[1]) // 2
    curr_rel_pos_on_sensor_width = float(corrected_pos) / sensor_size[1]
    expected_rel_pos = peak_expected_relative_location[2]

    # Calculate the correction factor in x
    sensor_center_dx = int((expected_rel_pos - curr_rel_pos_on_sensor_width) * sensor_size[1])

    # Use the vertical extent of the bar to re-center the sensor in y
    b_x0 = int(accepted_loc - accepted_width // 2)
    b_x = int(np.round(b_x0 + accepted_width))
    bar_bw = search_area_bw[:, b_x0:b_x]

    # Find the center of mass of the bar
    y, x = np.where(bar_bw)

    # Did we find the control bar?
    if len(y) == 0:

        # If the control bar was not found, just return the center of the
        # search area (with the expected size)
        y0 = sensor_center[0] - sensor_size[0] // 2
        y = y0 + sensor_size[0]
        x0 = sensor_center[1] - sensor_size[1] // 2
        x = x0 + sensor_size[1]

        # @ToDo Come up with a proper score
        sensor_score = 0.0

        return strip_gray_inverted[y0: y, x0: x], [y0, y, x0, x], sensor_score

    # Calculate the correction factor in y
    sensor_center_dy = int(np.round(np.mean(y)) - sensor_search_area[0] // 2)

    # New coordinates around the shifted center of the sensor
    y0 = sensor_center[0] + sensor_center_dy - sensor_size[0] // 2
    y = y0 + sensor_size[0]
    x0 = sensor_center[1] + sensor_center_dx - sensor_size[1] // 2
    x = x0 + sensor_size[1]

    # @ToDo Come up with a proper score
    sensor_score = 1.00

    # Return the extracted (inverted) sensor
    return strip_gray_inverted[y0: y, x0: x], [y0, y, x0, x], sensor_score


def get_sensor_contour_original(image, gray, sensor_center, force_output_shape=False,
                                output_shape=(40, 175), window=(120, 270),
                                border_margin=4,):

    # erode, thresh = threshold_image(gray, method=method)

    exp_x = sensor_center[1]
    exp_y = sensor_center[0]
    window_top_left = (round(exp_y - (window[0] / 2)), round(exp_x - (window[1] / 2)))

    gray_windowed = gray[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                               window[1]]
    e, t, score = threshold_image_original(gray_windowed, output_shape=output_shape)
    erode = np.zeros(gray.shape, np.uint8)
    thresh = np.zeros(gray.shape, np.uint8)
    erode[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                window[1]] = e
    thresh[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                 window[1]] = t
    sensor_area_masked_filled = thresh
    # # Get the contour of the sensor area based on the thresholded image
    # contour, img2 = find_central_contour(image, erode)
    #
    # # Mask the thresholded image using the contour
    # sensor_area_masked = np.zeros(gray.shape, np.uint8)
    # sensor_area_masked[contour[1]:contour[1] + contour[3], contour[0]:contour[0] + contour[2]] = erode[
    #                                                                                 contour[1]:contour[1] + contour[3],
    #                                                                                 contour[0]:contour[0] + contour[2]]
    # # Fill the sensor area and recompute contour
    # sensor_area_masked_filled = fill_mask(erode)
    # contour, _ = find_central_contour(image.copy(), sensor_area_masked_filled)
    contour, img2 = find_central_contour(image.copy(), sensor_area_masked_filled)

    if contour is None:
        return None, image, img2, erode, thresh, None, sensor_area_masked_filled, None

    # Control picked sensor area
    cv2.rectangle(image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (0, 255, 0), 1)
    # Add centroid of contour
    cv2.circle(image, (round(contour[0] + contour[2] / 2), round(contour[1] + contour[3] / 2)), 3, (255, 0, 0), -1)
    # Add centroid of poc test crop
    cv2.circle(image, (int(exp_x), int(exp_y)), 3, (0, 0, 255), -1)

    ret = [int(exp_x), int(exp_y), contour[0], contour[1], contour[2], contour[3]]

    if force_output_shape:

        detected_sensor_centroid = (round(contour[0] + (contour[2] / 2)),
                                    round(contour[1] + (contour[3] / 2)))

        if border_margin:
            output_shape = tuple(x + border_margin for x in output_shape)
        detected_sensor_top_left = (round(detected_sensor_centroid[0] - (output_shape[1] / 2)),
                                    round(detected_sensor_centroid[1] - (output_shape[0] / 2)))

        sensor = gray[detected_sensor_top_left[1]:detected_sensor_top_left[1] + output_shape[0],
                 detected_sensor_top_left[0]:detected_sensor_top_left[0] + output_shape[1],]
    else:
        sensor = gray[contour[1]:contour[1] + contour[3], contour[0]:contour[0] + contour[2]]

    sensor = invert_image(sensor)

    return ret, image, img2, erode, thresh, sensor, sensor_area_masked_filled, score


def get_sensor_contour(image, gray, method, force_output_shape=False, output_shape=(40, 175), window=(120, 270),
                       border_margin=4, sensor_center_offset=(0, 0)):

    # erode, thresh = threshold_image(gray, method=method)

    exp_x = gray.shape[1] / 2 + sensor_center_offset[1]
    exp_y = gray.shape[0] / 2 + sensor_center_offset[0]
    window_top_left = (round(exp_y - (window[0] / 2)), round(exp_x - (window[1] / 2)))

    gray_windowed = gray[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                               window[1]]
    e, t, score = threshold_image(gray_windowed, method=method, output_shape=output_shape,
                                  exp_center=(exp_x - window_top_left[1], exp_y - window_top_left[0]))
    erode = np.zeros(gray.shape, np.uint8)
    thresh = np.zeros(gray.shape, np.uint8)
    erode[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                window[1]] = e
    thresh[window_top_left[0]:window_top_left[0] + window[0], window_top_left[1]:window_top_left[1] +
                                                                                 window[1]] = t
    sensor_area_masked_filled = thresh
    # # Get the contour of the sensor area based on the thresholded image
    # contour, img2 = find_central_contour(image, erode)
    #
    # # Mask the thresholded image using the contour
    # sensor_area_masked = np.zeros(gray.shape, np.uint8)
    # sensor_area_masked[contour[1]:contour[1] + contour[3], contour[0]:contour[0] + contour[2]] = erode[
    #                                                                                 contour[1]:contour[1] + contour[3],
    #                                                                                 contour[0]:contour[0] + contour[2]]
    # # Fill the sensor area and recompute contour
    # sensor_area_masked_filled = fill_mask(erode)
    # contour, _ = find_central_contour(image.copy(), sensor_area_masked_filled)
    contour, img2 = find_central_contour(image.copy(), sensor_area_masked_filled)

    if contour is None:
        return None, image, img2, erode, thresh, None, sensor_area_masked_filled, None

    # Control picked sensor area
    cv2.rectangle(image, (contour[0], contour[1]), (contour[0] + contour[2], contour[1] + contour[3]), (0, 255, 0), 1)
    # Add centroid of contour
    cv2.circle(image, (round(contour[0] + contour[2] / 2), round(contour[1] + contour[3] / 2)), 3, (255, 0, 0), -1)
    # Add centroid of poc test crop
    cv2.circle(image, (int(exp_x), int(exp_y)), 3, (0, 0, 255), -1)

    ret = [int(exp_x), int(exp_y), contour[0], contour[1], contour[2], contour[3]]

    if force_output_shape:

        detected_sensor_centroid = (round(contour[0] + (contour[2] / 2)),
                                    round(contour[1] + (contour[3] / 2)))

        if border_margin:
            output_shape = tuple(x + border_margin for x in output_shape)
        detected_sensor_top_left = (round(detected_sensor_centroid[0] - (output_shape[1] / 2)),
                                    round(detected_sensor_centroid[1] - (output_shape[0] / 2)))

        sensor = gray[detected_sensor_top_left[1]:detected_sensor_top_left[1] + output_shape[0],
                 detected_sensor_top_left[0]:detected_sensor_top_left[0] + output_shape[1],]
    else:
        sensor = gray[contour[1]:contour[1] + contour[3], contour[0]:contour[0] + contour[2]]

    sensor = invert_image(sensor)

    return ret, image, img2, erode, thresh, sensor, sensor_area_masked_filled, score


def calculate_gradient_magnitude(gray, suppress_double_edge=False):
    """Calculates the gradient magnitude of a gray-value image."""

    # Compute the horizontal and vertical gradients
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Clean the gradients
    gradX = cv2.GaussianBlur(gradX, (1, 15), 3.0)
    gradY = cv2.GaussianBlur(gradY, (15, 1), 3.0)

    # Gradient magnitude
    G = np.sqrt(gradX ** 2 + gradY ** 2)

    G_half = None
    if suppress_double_edge:

        # Get rid of the double border
        gradX_half = gradX.copy()
        gradY_half = gradY.copy()
        gradX_half[gradX_half < 0] = 0
        gradY_half[gradY_half < 0] = 0
        G_half = np.sqrt(gradX_half ** 2 + gradY_half ** 2)

    return G, G_half

def enhance_image_advanced(img, clahe_clip_limit=1, clahe_tile_size=5, gaussian_sigma=3, std_filter_window=25):
    """Enhances the edges of a gray-value image."""

    # Make sure we work on a (copy of a) gray-scale image
    if img.ndim > 2:
        gray = BGR2Gray(img.copy())
    else:
        gray = img.copy()

    # Initialize a local contrast 
    clahe = None
    if clahe_clip_limit > 0 and clahe_tile_size > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(clahe_tile_size, clahe_tile_size))

    # Low-pass filter the image
    if gaussian_sigma > 0:
        sz = gaussian_sigma * 2 + 1
        gray = cv2.GaussianBlur(gray, (sz, sz), float(gaussian_sigma))

    # Run a standard deviation filter
    if std_filter_window > 0:
        gray = std_filter(gray, std_filter_window)

    # Enhance contrast
    gray = cv2.convertScaleAbs(gray)

    if clahe is not None:
        gray = clahe.apply(gray)

    # Return the result
    return gray


def enhance_image(gray, method=1):
    """Enhances the edges of a gray-value image using one of two methods.

    method=1: adds the gradient magnitude to the original image.
    method=1: multiplies the image by the normalized gradient magnitude.

    The returned image is scaled and has datatype np.uint8.
    """

    # Calculate the gradient magnitude
    G = calculate_gradient_magnitude(gray)[0]

    # Create a copy of the image with np.float data type
    gray_copy = gray.copy().astype(np.float)

    if method == 1:
        return cv2.convertScaleAbs(gray_copy + G)

    elif method == 2:
        mn = np.min(G)
        mx = np.max(G)
        G = (G - mn) / (mx - mn)
        return cv2.convertScaleAbs(G * gray_copy)

    else:
        raise Exception("Argument 'method' must be set either 1 or 2.")


def std_filter(img, sz=3):

    img32 = img.copy().astype(np.float32)

    mu = cv2.blur(img32, (sz, sz))
    mu2 = cv2.multiply(img32, img32)
    sigma = cv2.sqrt(mu2 - cv2.multiply(mu, mu))

    return sigma


def extract_rotated_strip_from_box(box_gray, box):
    """Segments the strip from the box image and rotates it so that it is horizontal."""

    # Segment using Li
    optimal_threshold = filters.threshold_li(box_gray)
    BW = box_gray > optimal_threshold

    # Clean up the mask
    BW = binary_fill_holes(BW)
    BW = binary_opening(BW, iterations=3)

    # Find the connected components
    labels, nb = label(BW)

    # Find the largest object
    areas = [np.sum(labels == x) for x in range(1, nb + 1)]
    indx = 1 + np.argmax(areas)

    # Copy the object to a new mask
    nBW = (0 * BW.copy()).astype(np.uint8)
    nBW[labels == indx] = 255

    # Find the (possibly rotated) contour
    # contours, hierarchy = cv2.findContours(nBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cv2.getVersionMajor() == 4:
        contours, hierarchy = cv2.findContours(
            nBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, hierarchy = cv2.findContours(
            nBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Make sure to work with the largest contour
    if len(contours) == 1:
        contour = contours[0]

    else:
        contour = None
        max_length = -1
        for c in contours:
            if c.shape[0] > max_length:
                max_length = c.shape[0]
                contour = c

    if contour is None:
        return None, None

    # Get the coarse orientation from the bounding box
    _, _, width, height = cv2.boundingRect(contour)

    # Get the rotated rectangle; the angle is not obvious to
    # interpret; we will need the bounding rectangle for that
    rect = cv2.minAreaRect(contour)

    # Apply the rotation (this assumes that the extracted rectangle has
    # its width somewhat parallel to the X axis)
    if width > height:
        angle = rect[2]
        if abs(angle) < 45:
            corr_angle = -1 * angle
        else:
            corr_angle = 90 + angle

        # Rotate
        nBW_rotated = rotate(nBW, corr_angle)
        box_gray_rotated = rotate(box_gray, corr_angle)
        box_rotated = rotate(box, corr_angle)

    else:

        # The strip appears to be oriented vertically.
        # This is most likely wrong; we won't try to
        # rotate it.
        nBW_rotated = nBW
        box_gray_rotated = box_gray
        box_rotated = box

    # Find the contour of the rotated BW mask
    # contours, _ = cv2.findContours(nBW_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cv2.getVersionMajor() == 4:
        contours, _ = cv2.findContours(
            nBW_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(
            nBW_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Make sure to work with the largest contour
    if len(contours) == 1:
        contour = contours[0]

    else:
        contour = None
        max_length = -1
        for c in contours:
            if c.shape[0] > max_length:
                max_length = c.shape[0]
                contour = c

    if contour is None:
        return None, None

    # Get the coarse orientation from the bounding box
    x, y, width, height = cv2.boundingRect(contour)

    # Get the bounding box closer to the rectangle
    y0, y, x0, x = adapt_bounding_box(nBW_rotated, x, y, width, height, fraction=0.75)

    # Extract the rotated strip
    strip_gray = box_gray_rotated[y0: y, x0: x]
    strip = box_rotated[y0: y, x0: x]

    # Return
    return strip_gray, strip


def adapt_bounding_box(bw, x0, y0, width, height, fraction=0.75):
    """Make the bounding box come closer to the strip by remove bumps along the outline."""

    # Make sure we have a binary mask
    bw = bw > 0

    # height, width = bw.shape
    fraction_h = fraction * height
    fraction_w = fraction * width

    new_x0 = x0
    new_x = x0 + width - 1
    new_y0 = y0
    new_y = y0 + height - 1

    # End points
    mid_x = x0 + width // 2 + 1
    mid_y = y0 + height // 2 + 1

    # From left
    for w in range(x0, mid_x):
        if np.sum(bw[:, w]) > fraction_h:
            new_x0 = w
            break

    # From right
    for w in range(x0 + width - 1, mid_x, -1):
        if np.sum(bw[:, w]) > fraction_h:
            new_x = w
            break

    # From top
    for h in range(y0, mid_y):
        if np.sum(bw[h, :]) > fraction_w:
            new_y0 = h
            break

    # From bottom
    for h in range(y0 + height - 1, mid_y, -1):
        if np.sum(bw[h, :]) > fraction_w:
            new_y = h
            break

    return new_y0, new_y, new_x0, new_x


def use_ocr_to_rotate_strip_if_needed(img_gray, img=None, text="COVID", on_right=True):
    """Try reading the given text on the strip. The text is expected to be on one
    side of the strip; if it is found on the other side, rotate the strip.

    We apply the same rotation also to the second image, if passed.
    """

    # If text is "", do nothing
    if text == "":
        return img_gray, img

    # Try with the following orientations
    angles = [0, -90, 90]

    # Use tesseract to read text from the strip. If successful,
    # this can be used to figure out the direction in which the
    # strip was placed under the camera. In a first attempt, we
    # search for the given text.

    for angle in angles:

        # Rotate the image
        rotated_img_gray = rotate(img_gray.copy(), angle)

        # Search for the text
        results = pytesseract.image_to_data(rotated_img_gray, output_type=Output.DICT)
        n_boxes = len(results['level'])
        for i in range(n_boxes):
            if text.upper() in results['text'][i].upper():

                center_of_mass_x = results['left'][i] + results['width'][i] // 2
                center_of_mass_y = results['top'][i] + results['height'][i] // 2

                # Found: now consider the possible cases
                if angle == 0:

                    # The image was not rotated; so it's still lying horizontally
                    if on_right:
                        if center_of_mass_x < rotated_img_gray.shape[1] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)
                    else:
                        if center_of_mass_x > rotated_img_gray.shape[1] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)

                    return img_gray, img

                elif angle == -90:

                    # The image was rotated 90 degrees clockwise; "right" is now "down"
                    if on_right:
                        if center_of_mass_y < rotated_img_gray.shape[0] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)
                    else:
                        if center_of_mass_y > rotated_img_gray.shape[0] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)

                    return img_gray, img

                else:

                    # The image was rotated 90 degrees counter-clockwise; "right" is now "up"
                    if on_right:
                        if center_of_mass_y > rotated_img_gray.shape[0] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)
                    else:
                        if center_of_mass_y < rotated_img_gray.shape[0] // 2:
                            # The label is on the wrong side of the strip
                            img_gray = rotate(img_gray, 180)
                            if img is not None:
                                img = rotate(img, 180)

                    return img_gray, img

    return img_gray, img


def read_patient_data_by_ocr(
        image,
        known_manufacturers=consts.KnownManufacturers):
    """Try to extract the patient data by OCR."""

    # Use a gray-value image (works better than RGB)
    image_gray = BGR2Gray(image)

    # Initialize outputs
    fid = ""
    manufacturer = ""

    # Try with different angles
    angles = [0, -90, 90, 180]

    for angle in angles:

        # Stop if we have everything
        if fid != "" and manufacturer != "":
            return fid, manufacturer

        # Use tesseract to read text from the strip. If successful,
        # this can be used to figure out the direction in which the
        # strip was placed under the camera. In a first attempt, we
        # search for the given text.
        try:
            results = pytesseract.image_to_data(
                rotate(image_gray.copy(), angle),
                output_type=Output.DICT
            )
        except:
            continue

        # Examine the results
        n_boxes = len(results['level'])
        for i in range(n_boxes):
            current_text = results['text'][i].upper()
            if current_text != "":

                # Test for manufacturer name
                if current_text.upper() in known_manufacturers:
                    manufacturer = current_text.upper()
                    continue

                # Test for fid
                match = re.search(r'^(?P<fid>[A-Z]+[0-9]{6,18}).*', current_text)
                if match is None:
                    continue
                else:
                    fid = match.group('fid')

    return fid, manufacturer

