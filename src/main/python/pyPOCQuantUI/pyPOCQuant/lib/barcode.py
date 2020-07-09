import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pyzbar.locations import Rect
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
from re import findall
from skimage import exposure
import math
import re
from typing import Union

from pyPOCQuant.lib.processing import BGR2Gray
from pyPOCQuant.lib.consts import SymbolTypes


class Barcode(object):
    """Pythonic barcode object."""

    def __init__(self, top: int, left: int, width: int, height: int, data: Union[bytes, str], symbol: str):
        """Constructor."""
        self.top = top
        self.left = left
        self.width = width
        self.height = height
        if type(data) is bytes:
            self.data = data.decode("utf-8")
        else:
            self.data = data
        self.symbol = symbol

    @classmethod
    def from_barcode(cls, barcode):
        """Initialize from pyzbar barcode object."""
        top = barcode.rect.top
        left = barcode.rect.left
        width = barcode.rect.width
        height = barcode.rect.height
        data = barcode.data
        symbol = barcode.type
        return cls(top, left, width, height, data, symbol)

    def scale(self, factor: float):
        """Scale the barcode object by given factor.

        The (top, left) is scaled accordingly.
        """
        self.top = int(factor * self.top)
        self.left = int(factor * self.left)
        self.width = int(factor * self.width)
        self.height = int(factor * self.height)

    def __str__(self):
        return f"Barcode of type '{self.symbol}' " \
               f"and data '{self.data}' at " \
               f"location ({self.left}, {self.top}) " \
               f"and size ({self.width}, {self.height})"

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"top={self.top}, left={self.left}, " \
               f"width={self.width}, height={self.height}, " \
               f"data='{self.data}', symbol='{self.symbol}')"


def detect(image: np.ndarray, expected_area=22000, expected_aspect_ratio=7.5, barcode_border=75, blur_size=(3, 3),
           morph_rect=(9, 3), mm_iter=1, qc=True, verbose=False):
    """Detect the barcode in the image.

    Adapted from: https://www.pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/

    Returns the extracted barcode image, the coordinates of the extracted rectangle,
    the (possibly rotated) image, and (if qc is True) a copy of the (possibly rotated)
    image with the extracted rectangle coordinates overlaid on it.
    """

    # Make sure the image is an array with three "channels"
    if type(image) is not np.ndarray:
        image = np.ndarray(image)

    if image.shape[2] != 3:
        raise Exception("RGB or BGR image expected.")

    # Run the extraction and check the orientation of the barcode and
    # the image. If necessary, rotate the image and rerun the extraction.
    y = 0
    x = 0
    h = image.shape[0]
    w = image.shape[1]
    box = [[x, y], [x, h], [y, w], [w, h]]
    correct_orientation = False
    num_rotations = 0
    give_up = False
    while correct_orientation is False:

        if num_rotations > 5:
            print(f"Giving up trying rotations.")
            give_up = True
            break

        if verbose:
            print(f"Rotations so far: {num_rotations}")

        # We start with a copy of the B/W image
        gray = BGR2Gray(image).copy()

        # Sharpen the image
        blurred = cv2.GaussianBlur(gray, (9, 9), 10.0)
        gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0, image)

        # Image main axes
        x_mid = gray.shape[1] / 2
        y_mid = gray.shape[0] / 2

        # Compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction using OpenCV
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # Subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # Blur and threshold the image
        blurred = cv2.blur(gradient, blur_size)
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # Construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_rect)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=mm_iter)
        closed = cv2.dilate(closed, None, iterations=mm_iter)

        # Find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts = cv2.findContours(
            closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = imutils.grab_contours(cnts)

        # Now do the same with a 90-degree rotated morphological rectangle
        kernel_rot = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_rect[1], morph_rect[0]))
        closed_rot = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_rot)

        # Perform a series of erosions and dilations
        closed_rot = cv2.erode(closed_rot, None, iterations=mm_iter)
        closed_rot = cv2.dilate(closed_rot, None, iterations=mm_iter)

        # Find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts_rot = cv2.findContours(
            closed_rot.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_rot = imutils.grab_contours(cnts_rot)

        # Process all contours in the two orientations, and pick the best

        c_areas = []
        c_aspect_ratios = []
        for current in c:
            area_c, aspect_ratio_c = calc_area_and_approx_aspect_ratio(current)
            c_areas.append(area_c - expected_area)
            c_aspect_ratios.append(aspect_ratio_c - expected_aspect_ratio)

        c_scores = []
        if len(c_areas) > 0:
            c_score_area = np.array(c_areas) / np.std(c_areas)
            c_score_aspect_ratio = np.array(
                c_aspect_ratios) / np.std(c_aspect_ratios)
            c_scores = np.sqrt(c_score_area ** 2 + c_score_aspect_ratio ** 2)
            if verbose:
                print(f"Min c score = {np.min(c_scores)}")

        c_rot_areas = []
        c_rot_aspect_ratios = []
        for current_rot in c_rot:
            area_c_rot, aspect_ratio_c_rot = calc_area_and_approx_aspect_ratio(
                current_rot)
            c_rot_areas.append(area_c_rot - expected_area)
            c_rot_aspect_ratios.append(
                aspect_ratio_c_rot - expected_aspect_ratio)

        c_rot_scores = []
        if len(c_rot_areas) > 0:
            c_rot_score_area = np.array(c_rot_areas) / np.std(c_rot_areas)
            c_rot_score_aspect_ratio = np.array(
                c_rot_aspect_ratios) / np.std(c_rot_aspect_ratios)
            c_rot_scores = np.sqrt(c_rot_score_area ** 2 +
                                   c_rot_score_aspect_ratio ** 2)
            if verbose:
                print(f"Min c_rot score = {np.min(c_rot_scores)}")

        # If both orientations failed to provide any contour,
        # there is no point to try with a 90-degree rotation.
        if len(c_scores) == 0 and len(c_rot_scores) == 0:
            return None, (0, 0, 0, 0), None, None

        # Find the orientation with the best score
        best_c_score_index = -1
        best_c_score = np.inf
        if len(c_scores) > 0:
            best_c_score_index = np.array(c_scores).argmin(axis=0)
            best_c_score = c_scores[best_c_score_index]

        best_c_rot_score_index = -1
        best_c_rot_score = np.inf
        if len(c_rot_scores) > 0:
            best_c_rot_score_index = np.array(c_rot_scores).argmin(axis=0)
            best_c_rot_score = c_rot_scores[best_c_rot_score_index]

        # Which direction of the filters gave the best response?
        if best_c_rot_score < best_c_score:
            # Rotate the original image by 90 degrees cw
            image = rotate(image, -90)
            correct_orientation = False
            num_rotations += 1
            if verbose:
                print(f"Rotated image by 90 degrees clockwise.")
            continue

        # If we are here, the c contours contain the best score.

        # Extract the winning contour
        c = c[best_c_score_index]

        # Inform
        if verbose:
            print(f"Best score = {best_c_score}")

        # Compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Extract the bar code
        x, y, w, h = cv2.boundingRect(box)

        # One more control to make sure the orientation is correct
        if w > h:

            # The image is either oriented correctly, or flipped 180 degrees
            # towards the bottom.
            if y < y_mid:

                # The image seems to be oriented correctly
                correct_orientation = True

            else:

                # The (original) image must be rotated 180 degrees
                image = rotate(image, 180)

                # Inform
                if verbose:
                    print(f"Rotated image by 180 degrees.")

                # Try again
                correct_orientation = False
                num_rotations += 1

                # Skip the rest
                continue

        else:

            # Try another rotation by 90 degrees cw
            image = rotate(image, -90)
            correct_orientation = False
            num_rotations += 1
            if verbose:
                print(f"Rotated image by 90 degrees clockwise.")
            continue

    if give_up is True:
        return None, (0, 0, 0, 0), None, None

    # Now extract the barcode image with a user-defined border around (and make B/W)
    b_y0 = y - barcode_border
    if b_y0 < 0:
        b_y0 = 0
    b_y = y + h + barcode_border
    if b_y > image.shape[0]:
        b_y = image.shape[0]
    b_x0 = x - barcode_border
    if b_x0 < 0:
        b_x0 = 0
    b_x = x + w + barcode_border
    if b_x > image.shape[1]:
        b_x = image.shape[1]
    barcode_img = image[b_y0: b_y, b_x0: b_x].copy()
    barcode_img = BGR2Gray(barcode_img)

    # Draw a bounding box around the detected barcode
    mask_image = None
    if qc:
        mask_image = image.copy()
        cv2.drawContours(mask_image, [box], -1, (0, 255, 0), 3)
        # plt.imshow(mask_image)
        # plt.show()

    # Return
    return barcode_img, (b_x0, b_y0, barcode_img.shape[1], barcode_img.shape[0]), image, mask_image


def rotate(image, angle):
    """Rotate the image by given angle in degrees."""

    if angle == 0:
        return image

    # Image size
    height, width = image.shape[:2]

    # Image center, for getRotationMatrix2D() in (x, y) order
    center = (width / 2, height / 2)

    # Get transformation matrix for the rotation with the given angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the size of the rotated bounding box
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # Subtract old image center and add new image center coordinates
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    # Now rotate with the calculated target image size
    return cv2.warpAffine(image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def calc_area_and_approx_aspect_ratio(contour):
    """Calculate area and approximate aspect ratio of a contour."""

    # Calculate area
    area = cv2.contourArea(contour)

    # Calculate aspect ratio
    x = []
    y = []
    for c in contour:
        x.append(c[0][0])
        y.append(c[0][1])

    dx = max(x) - min(x)
    dy = max(y) - min(y)
    aspect_ratio = dx / dy if dy > 0 else np.Inf

    return area, aspect_ratio


def rotate_90_if_needed(image):
    """Try to estimate the orientation of the image, and rotate if needed.

    @TODO: This is not very robust so far.
    """

    # We start with a copy of the B/W image
    gray = BGR2Gray(image).copy()

    # Blur
    blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Extract edges
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 90  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 250
    min_line_length = 500  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)

    v_votes = 0
    h_votes = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            if abs(x2 - x1) > abs(y2 - y1):
                h_votes += 1
            else:
                v_votes += 1

    plt.imshow(line_image)
    plt.show()

    if h_votes > v_votes:
        image = rotate(image, -90)

    return image


def read_FID_from_barcode_image(image):
    """
    Read the FID string from the barcode image using pytesseract and
    decode the barcode itself using pyzbar.
    """

    fid_tesseract = ""
    fid_pyzbar = ""

    if image is None:
        return fid_tesseract, fid_pyzbar

    # Use pytesseract to extract the FID
    text = pytesseract.image_to_string(image, lang='eng')
    fid = findall(r'\d{7}', text)
    if fid and len(fid) == 1:
        fid_tesseract = 'F' + fid[0]
    else:
        fid_tesseract = ""

    # Use pyzbar to decode the barcode
    decoded_objects = decode(image, SymbolTypes.TYPES.value)
    for obj in decoded_objects:
        fid_pyzbar = obj.data.decode("utf-8")

    # Give a score to the extraction (max is 3)
    if fid_tesseract == "" and fid_pyzbar == "":
        score = 0
    else:
        score = sum([
            fid_tesseract != "",
            fid_pyzbar != "",
            fid_tesseract == fid_pyzbar and fid_tesseract != ""])

    return fid_tesseract, fid_pyzbar, score


def get_fid_from_barcode_data(barcode_data, barcode_type="CODE128"):
    """Parse the output of pyzbar and retrieve the FID."""
    for barcode in barcode_data:
        if barcode.type == barcode_type:
            return barcode.data.decode("utf-8")
    return ""


def get_fid_from_box_image_using_ocr(box_img):
    """Use pytesseract to retrieve FID from the strip box image."""

    # Use pytesseract to extract the FID
    text = pytesseract.image_to_string(box_img, lang='eng')
    fid = findall(r'\d{7}', text)
    if fid and len(fid) == 1:
        fid_tesseract = fid[0]
    else:
        # Try rotating the image 90 degrees counter-clockwise
        box_img_90 = rotate(box_img, -90)
        text = pytesseract.image_to_string(box_img, lang='eng')
        fid = findall(r'\d{7}', text)
        if fid and len(fid) == 1:
            fid_tesseract = fid[0]
        else:
            fid_tesseract = ""

    return fid_tesseract


def try_extracting_barcode_with_rotation(image, angle_range=15, verbose=True, log_list: list=None):
    # Try with no rotation, if it works we stop.
    barcode_data = decode(image, SymbolTypes.TYPES.value)
    fid_pyzbar = get_fid_from_barcode_data(barcode_data)
    if fid_pyzbar != "":
        return fid_pyzbar, 0

    # Try with a range of positive rotations
    for angle in range(1, angle_range + 1):
        barcode_data = decode(rotate(image.copy(), angle), SymbolTypes.TYPES.value)
        fid_pyzbar = get_fid_from_barcode_data(barcode_data)
        if fid_pyzbar != "":
            if verbose:
                msg = f"Barcode found with a rotation of {angle} degrees."
                if log_list is None:
                    print(msg)
                else:
                    log_list.append(msg)
            return fid_pyzbar, angle, log_list

    # Try with a range of negative rotations
    for angle in range(angle_range, 0, -1):
        barcode_data = decode(rotate(image.copy(), angle), SymbolTypes.TYPES.value)
        fid_pyzbar = get_fid_from_barcode_data(barcode_data)
        if fid_pyzbar != "":
            if verbose:
                msg = f"Barcode found with a rotation of {angle} degrees."
                if log_list is None:
                    print(msg)
                else:
                    log_list.append(msg)
            return fid_pyzbar, angle, log_list

    return "", None, log_list


def find_strip_box_from_barcode_data_fh(image, barcode_data, qr_code_border=30, qc=False):
    """Extract the box around the strip using the QR barcode data."""

    if qc:
        qc_image = image.copy()
    else:
        qc_image = None

    # Initialize box coordinates
    all_y0 = []
    all_y = []
    all_x0 = []
    all_x = []

    # Keep track of the QR code width and height
    qr_code_widths = []
    qr_code_heights = []

    # Process the barcode data
    for barcode in barcode_data:
        if barcode.symbol == "QRCODE":
            if barcode.data.upper() == "BR":
                # Append candidate coordinates for bottom-rigth corner (x and y)
                current_x = barcode.left + barcode.width + qr_code_border
                current_y = barcode.top + barcode.height + qr_code_border
                all_x.append(current_x)
                all_y.append(current_y)
                qr_code_widths.append(barcode.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x, current_y), 11, (0, 0, 255), -1)

            elif barcode.data.upper() == "BL":
                # Append candidate coordinates for bottom-left corner (x0 and y)
                current_x0 = barcode.left - qr_code_border
                current_y = barcode.top + barcode.height + qr_code_border
                all_x0.append(current_x0)
                all_y.append(current_y)
                qr_code_widths.append(barcode.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x0, current_y), 11, (0, 0, 255), -1)

            elif barcode.data.upper() == "TR":
                # Append candidate coordinates for top-right corner (x and y0)
                current_x = barcode.left + barcode.width + qr_code_border
                current_y0 = barcode.top - qr_code_border
                all_x.append(current_x)
                all_y0.append(current_y0)
                qr_code_widths.append(barcode.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x, current_y0), 11, (0, 0, 255), -1)

            elif barcode.data.upper() == "TL":
                # Append candidate coordinates for top-left corner (x0 and y0)
                current_x0 = barcode.left - qr_code_border
                current_y0 = barcode.top - qr_code_border
                all_x0.append(current_x0)
                all_y0.append(current_y0)
                qr_code_widths.append(barcode.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x0, current_y0), 11, (0, 0, 255), -1)

            elif barcode.data.upper() == "TL_P":
                # @TODO: Use this to make sure that the page is oriented correctly.
                if qc:
                    qc_image = cv2.circle(qc_image, (barcode.left - qr_code_border, barcode.top - qr_code_border), 11,
                                          (0, 0, 255), -1)

            elif barcode.data.upper() == "R_G":
                # Currently ignored
                pass

            elif barcode.data.upper() == "L_G":
                # Currently ignored
                pass

            else:
                # This is assumed to be the barcode with patient data; we can ignore it.
                pass

        else:
            print(f"Unexpected barcode with type {barcode.type}.")

    # Now extract the box
    x0 = -1
    if len(all_x0) > 0:
        x0 = int(np.median(all_x0))
    x = -1
    if len(all_x) > 0:
        x = int(np.median(all_x))
    y0 = -1
    if len(all_y0) > 0:
        y0 = int(np.median(all_y0))
    y = -1
    if len(all_y) > 0:
        y = int(np.median(all_y))

    if x0 != -1 and x != -1 and x > x0 and y0 != -1 and y != -1 and y > y0:
        box = image[y0:y, x0:x]
        box_rect = [y0, y, x0, x]
        if qc:
            qc_image = qc_image[y0:y, x0:x, :]
    else:
        box = None
        box_rect = None

    # Calculate the size of the QR codes
    qr_code_width = 0 if len(qr_code_widths) == 0 else int(np.median(qr_code_widths))
    qr_code_height = 0 if len(qr_code_heights) == 0 else int(np.median(qr_code_heights))

    return box, (qr_code_width, qr_code_height), qc_image, box_rect


def find_strip_box_from_barcode_data(image, barcode_data, qr_code_border=30, qr_code_spacer=40, barcode_border=80,
                                     qc=False):
    """Extract the box around the strip using the QR barcode data."""

    if qc:
        qc_image = image.copy()
    else:
        qc_image = None

    # Initialize box coordinates
    all_y0 = []
    all_y = []
    all_x0 = []
    all_x = []

    # x coordinate of the left edge of the barcode
    x_barcode = -1

    # Keep track of the QR code width and height
    qr_code_widths = []
    qr_code_heights = []

    # Process the barcode data
    for barcode in barcode_data:
        if barcode.type == "QRCODE":
            if barcode.data.decode("utf-8").upper() == "BR":
                # Append candidate coordinates for bottom-rigth corner (x and y)
                current_x = barcode.rect.left + barcode.rect.width + qr_code_border
                current_y = barcode.rect.top + barcode.rect.height + qr_code_border
                all_x.append(current_x)
                all_y.append(current_y)
                qr_code_widths.append(barcode.rect.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.rect.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x, current_y), 11, (0, 0, 255), -1)

            elif barcode.data.decode("utf-8").upper() == "BL":
                # Append candidate coordinates for bottom-left corner (x0 and y)
                current_x0 = barcode.rect.left - qr_code_border
                current_y = barcode.rect.top + barcode.rect.height + qr_code_border
                all_x0.append(current_x0)
                all_y.append(current_y)
                qr_code_widths.append(barcode.rect.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.rect.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x0, current_y), 11, (0, 0, 255), -1)

            elif barcode.data.decode("utf-8").upper() == "TR":
                # Append candidate coordinates for top-right corner (x and y0)
                current_x = barcode.rect.left + barcode.rect.width + qr_code_border
                current_y0 = barcode.rect.top - qr_code_border
                all_x.append(current_x)
                all_y0.append(current_y0)
                qr_code_widths.append(barcode.rect.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.rect.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x, current_y0), 11, (0, 0, 255), -1)

            elif barcode.data.decode("utf-8").upper() == "TL":
                # Append candidate coordinates for top-left corner (x0 and y0)
                current_x0 = barcode.rect.left - qr_code_border
                current_y0 = barcode.rect.top - qr_code_border
                all_x0.append(current_x0)
                all_y0.append(current_y0)
                qr_code_widths.append(barcode.rect.width + 2 * qr_code_border)
                qr_code_heights.append(barcode.rect.height + 2 * qr_code_border)
                if qc:
                    qc_image = cv2.circle(qc_image, (current_x0, current_y0), 11, (0, 0, 255), -1)

            elif barcode.data.decode("utf-8").upper() == "TL_P":
                # @TODO: Use this to make sure that the page is oriented correctly.
                if qc:
                    qc_image = cv2.circle(qc_image,
                                          (barcode.rect.left - qr_code_border, barcode.rect.top - qr_code_border), 11,
                                          (0, 0, 255), -1)

            else:
                print(f"Unexpected QR code with data {barcode.data.decode('utf-8')}.")

        elif barcode.type == "CODE128":

            # Return the (x) coordinate of the left edge of the barcode rectangle.
            # We can use this to crop it away or mask it for alignment later.
            x_barcode = barcode.rect.left

            if qc:
                cv2.line(qc_image, (x_barcode, barcode.rect.top), (x_barcode, barcode.rect.top + barcode.rect.height),
                         (0, 255, 0), 2)

        else:
            print(f"Unexpected barcode with type {barcode.type}.")

    # Now extract the box
    x0 = -1
    if len(all_x0) > 0:
        x0 = int(np.median(all_x0))
    x = -1
    if len(all_x) > 0:
        x = int(np.median(all_x))
    y0 = -1
    if len(all_y0) > 0:
        y0 = int(np.median(all_y0))
    y = -1
    if len(all_y) > 0:
        y = int(np.median(all_y))

    if x0 != -1 and x != -1 and y0 != -1 and y != -1:
        box = image[y0:y, x0:x]
        if qc:
            qc_image = qc_image[y0:y, x0:x, :]
    else:
        box = None

    # Calculate the size of the QR codes
    qr_code_width = int(np.median(qr_code_widths))
    qr_code_height = int(np.median(qr_code_heights))

    # Express x_barcode (without border) as a function of x_0
    if x_barcode != -1:
        x_barcode -= (barcode_border + qr_code_spacer + x0)

    return box, x_barcode, (qr_code_width, qr_code_height), qc_image


def try_extracting_barcode_with_linear_stretch(image, lower_bound_range=(25,), upper_bound_range=(98,)):
    # NOTE:  CONTRAST is KEY. Rescaling intensity a bit helps not only in detecting the barcode but also QR
    # codes. We might try other options such as Adaptive Hist, CLAHE, etc
    # NOTE2: Orientation might play a role - however minor. Prefered orientation for the barcode detector seams
    # horizontal but vertical works too

    gray = BGR2Gray(image.copy())

    for lb in lower_bound_range:
        for ub in upper_bound_range:

            # Linearly stretch the contrast
            pLb, pUb = np.percentile(gray, (lb, ub))
            stretched_gray = exposure.rescale_intensity(gray, in_range=(pLb, pUb))

            # Run the barcode detection
            barcode_data = decode(stretched_gray, SymbolTypes.TYPES.value)

            # Retrieve the FID from the barcode data
            fid_pyzbar = get_fid_from_barcode_data(barcode_data)

            if fid_pyzbar != "":
                return fid_pyzbar, stretched_gray

    return "", gray


def try_extracting_fid_and_all_barcodes_with_linear_stretch_fh(
        image,
        lower_bound_range=(0, 5, 15, 25, 35),
        upper_bound_range=(100, 98, 95, 92, 89),
        scaling=(1.0, )
    ):

    if image.ndim == 3:
        gray = BGR2Gray(image.copy())
    else:
        gray = image.copy()

    best_score = -1
    best_scaling_factor = 1.0
    best_barcode_data = None
    best_lb = 0
    best_ub = 100
    fid = ""
    manufacturer = ""
    plate = ""
    well = ""
    user = ""

    for scaling_factor in scaling:

        if scaling_factor != 1.0:
            w = int(scaling_factor * gray.shape[1])
            h = int(scaling_factor * gray.shape[0])
            gray_resized = cv2.resize(gray, (w, h), cv2.INTER_LANCZOS4)
        inv_scaling_factor = 1.0 / scaling_factor

        for lb in lower_bound_range:
            for ub in upper_bound_range:

                # Restart from the original contrast in the scaled image
                gray_process = gray_resized.copy()

                # Keep score
                score = 0

                # Linearly stretch the contrast
                pLb, pUb = np.percentile(gray_process, (lb, ub))
                stretched_gray = exposure.rescale_intensity(gray_process, in_range=(pLb, pUb))

                # Run the barcode detection
                barcode_data = decode(stretched_gray, [ZBarSymbol.QRCODE])

                # Are all QR codes and barcodes found successfully?
                for barcode in barcode_data:
                    if barcode.type == "QRCODE":
                        if barcode.data.decode("utf-8").upper() == "BR":
                            score += 1
                        elif barcode.data.decode("utf-8").upper() == "BL":
                            score += 1
                        elif barcode.data.decode("utf-8").upper() == "TR":
                            score += 1
                        elif barcode.data.decode("utf-8").upper() == "TL":
                            score += 1
                        elif barcode.data.decode("utf-8").upper() == "TL_P":
                            score += 1
                        elif barcode.data.decode("utf-8").upper() == "L_G":
                            # L_G QR code currently ignored and does not contribute to the score.
                            pass
                        elif barcode.data.decode("utf-8").upper() == "R_G":
                            # R_G QR code currently ignored and does not contribute to the score
                            pass
                        else:
                            # Try extracting the FID

                            match = re.search(
                                r'^(?P<fid>[A-Z]+[0-9]{6,18})-(?P<manufacturer>.+)-Plate (?P<plate>\d{1,3})-Well (?P<well>.+)-(?P<user>.+)$',
                                barcode.data.decode('utf-8'))
                            if match is None:
                                print(f"Unexpected QR code with data {barcode.data.decode('utf-8')}.")

                            else:
                                fid = match.group('fid')
                                manufacturer = match.group('manufacturer')
                                plate = match.group('plate')
                                well = match.group('well')
                                user = match.group('user')
                                score += 1
                    else:
                        print(f"Unexpected barcode type {barcode.type}.")

                if score == 6:

                    # Return a list of (scaled) Barcode objects
                    barcodes = []
                    for barcode in barcode_data:
                        try:
                            obj = Barcode.from_barcode(barcode)
                            obj.scale(inv_scaling_factor)
                            barcodes.append(obj)
                        except:
                            pass

                    return barcodes, fid, manufacturer, plate, well, user, lb, ub, score, scaling_factor

                else:
                    if score > best_score:
                        best_score = score
                        best_barcode_data = barcode_data
                        best_lb = lb
                        best_ub = ub
                        best_stretched_gray = stretched_gray
                        best_scaling_factor = scaling_factor

    # Return a list of (scaled) Barcode objects
    barcodes = []
    for barcode in best_barcode_data:
        try:
            obj = Barcode.from_barcode(barcode)
            obj.scale(1.0 / best_scaling_factor)
            barcodes.append(obj)
        except:
            pass

    return barcodes, fid, manufacturer, plate, well, user, best_lb, best_ub, best_score, best_scaling_factor


def try_extracting_all_barcodes_with_linear_stretch(
        image,
        lower_bound_range=(0, 5, 15, 25, 35),
        upper_bound_range=(100, 98, 95, 92, 89)
):
    # NOTE:  CONTRAST is KEY. Rescaling intensity a bit helps not only in detecting the barcode but also QR
    # codes. We might try other options such as Adaptive Hist, CLAHE, etc
    # NOTE2: Orientation might play a role - however minor. Preferred orientation for the barcode detector sems
    # horizontal but vertical works too

    if image.ndim == 3:
        gray = BGR2Gray(image.copy())
    else:
        gray = image.copy()

    best_score = -1
    best_barcode_data = None
    best_lb = 0
    best_ub = 100

    for lb in lower_bound_range:
        for ub in upper_bound_range:

            # result = [TL_P_found, TL_found, TR_found, BL_found, BR_found, FID_found]
            TL_P, TL, TR, BL, BR, FID = 0, 1, 2, 3, 4, 5
            result = [False, False, False, False, False, False]

            # Linearly stretch the contrast
            pLb, pUb = np.percentile(gray, (lb, ub))
            stretched_gray = exposure.rescale_intensity(gray, in_range=(pLb, pUb))

            # Run the barcode detection
            barcode_data = decode(stretched_gray, SymbolTypes.TYPES.value)

            # Are all QR codes and barcodes found successfully?
            for barcode in barcode_data:
                if barcode.type == "QRCODE":
                    if barcode.data.decode("utf-8").upper() == "BR":
                        result[BR] = True
                    elif barcode.data.decode("utf-8").upper() == "BL":
                        result[BL] = True
                    elif barcode.data.decode("utf-8").upper() == "TR":
                        result[TR] = True
                    elif barcode.data.decode("utf-8").upper() == "TL":
                        result[TL] = True
                    elif barcode.data.decode("utf-8").upper() == "TL_P":
                        result[TL_P] = True
                    else:
                        print(f"Unexpected QR code with data {barcode.data.decode('utf-8')}.")
                elif barcode.type == "CODE128":
                    # Let's check if the FID was read
                    if barcode.data.decode("utf-8") != "":
                        result[FID] = True
                else:
                    print(f"Unexpected barcode type {barcode.type}.")

            score = np.sum(result)
            if score == 6:
                return barcode_data, lb, ub, score
            else:
                if score > best_score:
                    best_score = score
                    best_barcode_data = barcode_data
                    best_lb = lb
                    best_ub = ub
                    best_stretched_gray = stretched_gray

    return best_barcode_data, best_lb, best_ub, best_score


def rotate_if_needed_fh(image, barcode_data, image_log, verbose=True):
    """Rotate the image if the orientation is not the expected one.

    Return tuple (image_was_rotated: boolean, image)
    """

    positions = {
        "TL_P": None,
        "TL": None,
        "TR": None,
        "BL": None,
        "BR": None,
        "L_G": None,
        "R_G": None
    }

    # Extract the information
    for barcode in barcode_data:
        if barcode.symbol == "QRCODE":
            key = barcode.data.upper()
            if key in positions:
                try:
                    positions[key] = {"x": barcode.left, "y": barcode.top}
                except:
                    positions[key] = None

    # Use the information to understand the orientation of the image

    # Case 1: correct orientation
    top_left_y = -1
    top_left_x = -1
    left_y = -1
    left_x = -1
    right_y = -1
    right_x = -1

    if positions["TL_P"] is not None:
        top_left_y = positions["TL_P"]["y"]
        top_left_x = positions["TL_P"]["x"]

    if positions["TL"] is not None:
        if top_left_y == -1:
            top_left_y = positions["TL"]["y"]
            top_left_x = positions["TL"]["x"]
        else:
            left_y = positions["TL"]["y"]
            left_x = positions["TL"]["x"]

    if positions["BL"] is not None:
        if top_left_y == -1:
            top_left_y = positions["BL"]["y"]
            top_left_x = positions["BL"]["x"]
        if left_y == -1:
            left_y = positions["BL"]["y"]
            left_x = positions["BL"]["x"]

    if positions["L_G"] is not None:
        if top_left_y == -1:
            top_left_y = positions["L_G"]["y"]
            top_left_x = positions["L_G"]["x"]
        if left_y == -1:
            left_y = positions["L_G"]["y"]
            left_x = positions["L_G"]["x"]

    if positions["TR"] is not None:
        right_y = positions["TR"]["y"]
        right_x = positions["TR"]["x"]

    if positions["BR"] is not None:
        if right_y == -1:
            right_y = positions["BR"]["y"]
            right_x = positions["BR"]["x"]

    if positions["R_G"] is not None:
        if right_y == -1:
            right_y = positions["R_G"]["y"]
            right_x = positions["R_G"]["x"]

    # Case 1: the image is already oriented correctly
    if top_left_y != -1 and left_y != -1 and left_x != -1 and right_x != -1:
        if top_left_y < left_y and left_x < right_x:
            # The image does not need to be rotated
            image_was_rotated = False
            return image_was_rotated, image, image_log

    # Case 2: the image is rotated 180 degrees
    if top_left_y != -1 and left_y != -1 and left_x != -1 and right_x != -1:
        if top_left_y > left_y and left_x > right_x:
            # The image needs to be rotated 180 degrees
            image = rotate(image, 180)
            image_was_rotated = True
            return image_was_rotated, image, image_log

    # Case 3: the image is rotated 90 degrees clockwise
    if left_y != -1 and left_x != -1 and left_y != -1 and right_y != -1:
        if left_x < top_left_x and left_y < right_y:
            # The image needs to be rotated 90 degrees counter-clockwise
            image = rotate(image, 90)
            image_was_rotated = True
            return image_was_rotated, image, image_log

    # Case 4: the image is rotated 90 degrees counter-clockwise
    if left_y != -1 and left_x != -1 and left_y != -1 and right_y != -1:
        if left_x > top_left_x and left_y > right_y:
            # The image needs to be rotated 90 degrees clockwise
            image = rotate(image, -90)
            image_was_rotated = True
            return image_was_rotated, image, image_log

    # Unhandled case!
    image_log.append(
        f"top_left = (y={top_left_y}, x={top_left_x}),"
        f" left = (y={left_y}, x={left_x}),"
        f" right = (y={right_y}, x={right_x})"
    )

    return False, image, image_log


def rotate_if_needed(image, barcode_data, image_log, verbose=True):
    """Rotate the image if the orientation is not the expected one.

    Return tuple (image_was_rotated: boolean, image)
    """

    # Center axes of the image
    mid_y = image.shape[0] // 2
    mid_x = image.shape[1] // 2

    # The barcode TL_P should be at the top left of the page
    x0 = -1
    y0 = -1
    for barcode in barcode_data:
        if barcode.type == "QRCODE":
            if barcode.data.decode("utf-8").upper() == "TL_P":
                x0 = barcode.rect.left
                y0 = barcode.rect.top
                break

    if x0 != -1 and y0 != -1:

        if verbose:
            # print(f"Barcode 'TL_P' found at location (x={x0}, y={y0})")
            # print(f"Image size is (h={image.shape[0]}, w={image.shape[1]})")
            image_log.append(f"Barcode 'TL_P' found at location (x={x0}, y={y0})")
            image_log.append(f"Barcode 'TL_P' found at location (x={x0}, y={y0})")

        image_was_rotated = False

        # We expect image height to be > image width
        if image.shape[0] > image.shape[1]:

            # The image is vertical, let's check if the TL_P QR code is at
            # the top-left or at the bottom-right corner of the image
            if x0 < mid_x and y0 < mid_y:

                # The TL_P is in the top-left quadrant, nothing to do
                image_was_rotated = False

            elif x0 > mid_x and y0 > mid_y:

                # The TL_P is in the bottom-right quadrant, rotate 180 degrees
                image = rotate(image, 180)

                image_was_rotated = True

            else:

                raise Exception("This case must still be handled!")

        else:

            # The image is horizontal, let's check if the TL_P QR code is at
            # the top-right or at the bottom-left corner of the image
            if x0 > mid_x // 2 and y0 < mid_y:

                # The TL_P is in the top-right quadrant, rotate 90 degrees counter-clockwise
                image = rotate(image, 90)

                image_was_rotated = True

            elif x0 < mid_x and y0 > mid_y:

                # The TL_P is in the bottom-left quadrant, rotate 90 degrees clockwise
                image = rotate(image, -90)

                image_was_rotated = True

            else:

                raise Exception("This case must still be handled!")

    else:

        # The TL_P QR code could not be found, try using the others.
        # From all the detected ones, try to estimate the center of the strip box.
        all_x = []
        all_y = []
        # Process the barcode data
        for barcode in barcode_data:
            if barcode.type == "QRCODE":
                if barcode.data.decode("utf-8").upper() == "BR" or \
                        barcode.data.decode("utf-8").upper() == "BL" or \
                        barcode.data.decode("utf-8").upper() == "TR" or \
                        barcode.data.decode("utf-8").upper() == "TL":
                    all_x.append(barcode.rect.left + barcode.rect.width // 2)
                    all_y.append(barcode.rect.top + barcode.rect.height // 2)

        x_strip = np.mean([np.min(all_x), np.max(all_x)])
        y_strip = np.mean([np.min(all_y), np.max(all_y)])

        d_x = x_strip - mid_x
        d_y = y_strip - mid_y

        if image.shape[0] < image.shape[1]:
            # The image is horizontal

            # Consistency check
            if abs(d_x) <= abs(d_y):
                print("Unexpected strip position!")
                image_was_rotated = False

                return image_was_rotated, image, image_log

            if d_x < 0:
                # Rotate 90 degrees counter-clockwise
                image = rotate(image, 90)
                image_was_rotated = True
            else:
                # Rotate 90 degrees clockwise
                image = rotate(image, -90)
                image_was_rotated = True
        else:
            # The image is vertical

            # Consistency check
            if abs(d_y) <= abs(d_x):
                print("Unexpected strip position!")
                image_was_rotated = False

                return image_was_rotated, image, image_log

            if d_y < 0:
                # Rotate 180 degrees
                image = rotate(image, 180)
                image_was_rotated = True
            else:
                # The image is in the right orientation
                image_was_rotated = False

    return image_was_rotated, image, image_log


def pick_FID_from_candidates(fid_pyzbar, fid_tesseract):
    if fid_pyzbar == "" and fid_tesseract == "":
        return "", 0

    # Give a score to the extraction (max is 3)
    score = sum([
        fid_tesseract != "",
        fid_pyzbar != "",
        fid_tesseract == fid_pyzbar and fid_tesseract != ""])

    # Now pick the FID
    if fid_pyzbar != "":

        # If fid_pyzbar is not "", we pick it
        fid = fid_pyzbar

    else:

        # If fid_pyzbar is "", we pick fid_tesseract, which
        # could be "" -- but then the score would be 0.
        fid = fid_tesseract

    return fid, score


def mask_strip(strip_gray, x_barcode, qr_code_extents):
    """Hide the barcode on the strip image."""
    strip_gray_masked = strip_gray.copy()
    rel_x_barcode = x_barcode - qr_code_extents[1]
    background_value = np.median(strip_gray_masked[:, rel_x_barcode - 5:rel_x_barcode])
    strip_gray_masked[:, rel_x_barcode:] = background_value
    return strip_gray_masked, background_value


def extract_strip_from_box(box, qr_code_width, qr_code_height, qr_code_spacer=40, slack=0):
    """Extract the strip from the strip box.

    qr_code_width: width ot the QR code
    qr_code_height: height ot the QR code
    qr_code_spacer: horizontal and vertical distance between the internal edge of the QR codes
                    and the beginning of the strip.
    slack: some buffer (subtracted from qr_code_spacer) to avoid cropping into the strip
    """
    vertical_offset = qr_code_height + qr_code_spacer - slack
    horizontal_offset = qr_code_width + qr_code_spacer - slack
    if box.ndim == 3:
        strip = box[vertical_offset:-vertical_offset, horizontal_offset:-horizontal_offset, :]
    else:
        strip = box[vertical_offset:-vertical_offset, horizontal_offset:-horizontal_offset]
    return strip


def get_fid_numeric_value_fh(fid):
    """Return the numeric value of the FID (as string).

    A FID could be in the form 'F0123456'. We want to preserve
    the leading 0 after we removed the 'F'.
    """
    if fid is None:
        return ""
    return ''.join(filter(lambda i: i.isdigit(), fid))


def get_fid_numeric_value(fid):
    """Return the numeric value of the FID."""
    if fid is None:
        return -1
    filtered_fid = ''.join(filter(lambda i: i.isdigit(), fid))
    if filtered_fid == '':
        return -1
    return int(filtered_fid)


def get_box_rotation_angle(pt1, pt2, pt3):
    v1_angle = np.arctan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))
    v2_angle = np.arctan2((pt3[1] - pt1[1]), (pt3[0] - pt1[0]))
    rot_angle = math.degrees(v2_angle - v1_angle)
    return rot_angle


def align_box_with_image_border_fh(barcode_data, image):
    qr_centroids = {}
    for code in barcode_data:
        qr_name = code.data
        qr_centroids[qr_name.replace('b', '')] = (int(code.left + (code.width / 2)),
                                                  int(code.top + (code.height / 2)))

    # Case we have BL and BR
    if {"BL", "BR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['BL'], (qr_centroids['BR'][0], qr_centroids['BL'][1]),
                                       qr_centroids['BR'])
        image_rotated = imutils.rotate_bound(image, -angle)
    # Case we have TL and TR
    elif {"TL", "TR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TL'], (qr_centroids['TR'][0], qr_centroids['TL'][1]),
                                       qr_centroids['TR'])
        image_rotated = imutils.rotate_bound(image, -angle)
    # Case we have TL and BL
    elif {"TL", "BL"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TL'], (qr_centroids['BL'][0], qr_centroids['TL'][1]),
                                       qr_centroids['BL'])
        image_rotated = imutils.rotate_bound(image, -(90 - abs(angle)))
    # Case we have TR and BR
    elif {"TR", "BR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TR'], (qr_centroids['BR'][0], qr_centroids['TR'][1]),
                                       qr_centroids['BR'])
        image_rotated = imutils.rotate_bound(image, -(90 - abs(angle)))
    # Case no valid pair was detected
    else:
        # Return same image
        image_rotated = image
        angle = -1

    return image_rotated, angle


def align_box_with_image_border(barcode_data, image):
    qr_centroids = {}
    for code in barcode_data:
        qr_name = code.data.decode()
        qr_centroids[qr_name.replace('b', '')] = (int(code.rect.left + (code.rect.width / 2)),
                                                  int(code.rect.top + (code.rect.height / 2)))

    # Case we have BL and BR
    if {"BL", "BR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['BL'], (qr_centroids['BR'][0], qr_centroids['BL'][1]),
                                       qr_centroids['BR'])
        image_rotated = imutils.rotate_bound(image, -angle)
    # Case we have TL and TR
    elif {"TL", "TR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TL'], (qr_centroids['TR'][0], qr_centroids['TL'][1]),
                                       qr_centroids['TR'])
        image_rotated = imutils.rotate_bound(image, -angle)
    # Case we have TL and BL
    elif {"TL", "BL"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TL'], (qr_centroids['BL'][0], qr_centroids['TL'][1]),
                                       qr_centroids['BL'])
        image_rotated = imutils.rotate_bound(image, -(90 - abs(angle)))
    # Case we have TR and BR
    elif {"TR", "BR"}.issubset(qr_centroids):
        angle = get_box_rotation_angle(qr_centroids['TR'], (qr_centroids['BR'][0], qr_centroids['TR'][1]),
                                       qr_centroids['BR'])
        image_rotated = imutils.rotate_bound(image, -(90 - abs(angle)))
    # Case no valid pair was detected
    else:
        # Return same image
        image_rotated = image
        angle = -1

    return image_rotated, angle
