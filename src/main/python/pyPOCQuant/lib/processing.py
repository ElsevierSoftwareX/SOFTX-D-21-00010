import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import scipy.signal
import cv2


def phase_only_correlation(in1: np.ndarray, in2: np.ndarray) -> np.ndarray:
    """Calculate phase-only correlation of two numpy arrays.

    :param in1: 2D numpy array.
    :param in2: 2D numpy array.
    :return: 2D np.float64 numpy array.
    """

    # Make sure the inputs are in np.float64 precision
    in1 = in1.astype(np.float64)
    in2 = in2.astype(np.float64)

    # Get and check image dimensions
    s_y1, s_x1 = in1.shape
    s_y2, s_x2 = in2.shape

    if s_y1 != s_y2 or s_x1 != s_x2:
        raise Exception('The images must be of the same size!')

    # Apply windowing
    taper = np.dot(np.blackman(s_y1).reshape(-1, 1),
                   np.blackman(s_x1).reshape(1, -1))

    in1 = in1 * taper
    in2 = in2 * taper

    # Calculate the Fourier transforms
    f1 = fft2(in1)
    f2 = fft2(in2)

    # Calculate the conjugate of f1
    cf1 = np.conj(f1)

    # Phase correlation
    fr = cf1 * f2
    f_corr = fftshift(ifft2(fr / np.abs(fr)))

    # Return the phase correlation
    return np.abs(f_corr)


def find_position_in_image_using_phase_corr(in1: np.ndarray, in2: np.ndarray) -> tuple:
    """Uses phase-only correlation to find the coordinates in `in2` where `in1` can be found.

    :param in1: 2D numpy array (must be strictly smaller, i.e. completely contained) in in2.
    :param in2: 2D numpy array.
    :return: tuple with (y = row, x = column) location of the center of in1 in in2.
    """

    # Get and check image dimensions
    s_y1, s_x1 = in1.shape
    s_y2, s_x2 = in2.shape

    # in1 must strictly be contained in in2
    if s_y1 >= s_y2 or s_x1 >= s_x2:
        raise Exception('Image in1 must strictly be contained in in2!')

    # Place in1 in the center of an enlarged version of itself that has the same size as in2
    e_in1 = in2.copy() * 0.0
    b_y1 = (e_in1.shape[0] // 2) - (s_y1 // 2)
    b_x1 = (e_in1.shape[1] // 2) - (s_x1 // 2)
    e_in1[b_y1:b_y1 + s_y1, b_x1:b_x1 + s_x1] = in1

    # Calculate the phase-only correlation
    f_corr = phase_only_correlation(e_in1, in2)

    # Find the coordinates of the maximum
    y, x = np.unravel_index(np.argmax(f_corr), f_corr.shape)

    # Return them
    return y, x


def find_position_in_image_using_norm_xcorr(in1: np.ndarray, in2: np.ndarray) -> tuple:
    """Uses normalized cross-correlation to find the coordinates in `in2` where `in1` can be found.

    :param in1: 2D numpy array (must be strictly smaller, i.e. completely contained) in in2.
    :param in2: 2D numpy array.
    :return: tuple with (y = row, x = column) location of the center of in1 in in2.
    """

    # Get and check image dimensions
    s_y1, s_x1 = in1.shape
    s_y2, s_x2 = in2.shape

    # in1 must strictly be contained in in2
    if s_y1 >= s_y2 or s_x1 >= s_x2:
        raise Exception('Image in1 must strictly be contained in in2!')

    # Calculate normalized cross-correlation using correlate2D from scipy.signal
    c_corr = scipy.signal.correlate2d(in2.astype(np.float64), in1.astype(
        np.float64), boundary='symm', mode='same')

    # Find the match
    y, x = np.unravel_index(np.argmax(c_corr), c_corr.shape)

    # Return them
    return y, x


def correlation_coefficient(image_1, image_2):
    """Create the normalized correlation coefficient (scalar) of two images."""
    if image_1.shape[0] != image_2.shape[0] or image_1.shape[1] != image_2.shape[1]:
        raise Exception(
            "`image_1` and `image_2` must have  the same dimensions!")

    product = np.mean((image_1 - image_1.mean()) * (image_2 - image_2.mean()))
    stds = image_1.std() * image_2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


def crop_image_around_position_to_size(image, y, x, size):
    """Crop an image to given size centered at coordinates (y, x).

    If the original image is too small, a cropped version will be returned.
    """

    half_size_y = size[0] // 2
    half_size_x = size[1] // 2

    in_image_y0 = y - half_size_y
    in_image_x0 = x - half_size_x
    in_image_y = y + half_size_y
    in_image_x = x + half_size_x

    if in_image_y0 < 0:
        in_image_y0 = 0

    if in_image_x0 < 0:
        in_image_x0 = 0

    if in_image_y >= image.shape[0]:
        in_image_y = image.shape[0]

    if in_image_x >= image.shape[1]:
        in_image_x = image.shape[1]

    out = image[in_image_y0:in_image_y, in_image_x0:in_image_x]

    if out.shape[0] < size[0] or out.shape[1] < size[1]:
        extended_out = np.zeros(size, dtype=image.dtype)
        dy = (size[0] - out.shape[0]) // 2
        dx = (size[1] - out.shape[1]) // 2
        extended_out[dy:dy + out.shape[0], dx:dx + out.shape[1]] = out
        return extended_out

    else:
        return out


def create_rgb_image(red, green, blue=None):
    """Merge three single channels into an RGB image."""

    if blue is None:
        blue = np.zeros(red.shape, dtype=red.dtype)

    view = np.dstack((red, green, blue))
    return view


def find_features(image, detector="surf", num_features=1000, hessian_threshold=10, use_latch_descriptor=False):
    """Find features in of a template inside a larger image."""

     # Initialize feature detector
    if detector == "surf":
        detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold)
    elif detector == "sift":
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
    elif detector == "orb":
        detector = cv2.ORB_create(nfeatures=num_features)
    elif detector == "kaze":
        detector = cv2.KAZE_create()
    elif detector == "akaze":
        detector = cv2.AKAZE_create()
    elif detector == "brisk":
        detector = cv2.BRISK_create()
    else:
        raise Exception("Unsupported detector.")

    if use_latch_descriptor:

        # Instantiate LATCH descriptor
        latch = cv2.xfeatures2d.LATCH_create(64, True, 15)

        # Detect
        kp = detector.detect(image)

        # Compute
        _, des = latch.compute(image, kp)

    else:

        kp, des = detector.detectAndCompute(image, None)

    return kp, des

def find_position_of_template_in_image_using_descriptors(template_kps, template_des, image_kps, image_des, template_size):
    """Find the template in the image using the extracted feature descriptors."""

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(template_des, image_des)

    # Sort matches by score (distance)
    matches = sorted(matches, key=lambda x: x.distance)

    # Get and cast the points to be used for transformation estimation
    s_pts = np.float32([template_kps[m.queryIdx].pt for m in matches])
    d_pts = np.float32([image_kps[m.trainIdx].pt for m in matches])

    # Estimate the rigid transformation
    M, mask = cv2.estimateAffinePartial2D(s_pts, d_pts, cv2.RANSAC)

    # Now project the coordinate of the center of the template onto the image
    x_m = template_size[1] / 2
    y_m = template_size[0] / 2
    src = np.array([[[x_m, y_m]]], dtype = np.float32)
    dest = np.squeeze(cv2.transform(src, M))

    # Return the coordinates (as indices)
    return int(dest[1]), int(dest[0])


def register_images_opencv_features(source, target, detector="surf", use_latch_descriptor=False,
                                    perspective=True, affine=False, rigid=False, num_features=1000,
                                    hessian_threshold=10, control_image=False):
    """Register 2 images using image features.

    Keyword arguments:
    :param source: source image to be registered (must be grayscale)
    :param target: target image (must be grayscale)
    :param detector: one of "orb", "kaze", "akaze", "brisk", "surf", "sift" (default if surf)
    :param use_latch_descriptor: True to use the new LATCH descriptor (requires openCV 3.1), False to use the
                                 default descriptors provided by the detectors (default is False)
    :param num_features: number of features (used only by the "orb" and "sift" detectors, default is 1000)
    :param hessian_threshold: threshold of the hessian of the images (used only by the "surf" detector, default is 10)
    :param perspective: register the image using a perspective transformation (optional, default=True).
    :param affine: register the image using an affine transformation (optional, default=False).
    :param rigid: register the image using a rigid transformation (optional, default=False).
    :param control_image: set to True to create a quality control image (default is False).
    :return: (aligned: aligned image,
              M : transformation matrix,
              mask: mask returned by cv2.findHomography(),
              view: quality control image,
              source_descr: list of source descriptors,
              target_descr: list of target descriptors).
    """

    # Initialize feature detector
    if detector == "surf":
        detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold)
    elif detector == "sift":
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=num_features)
    elif detector == "orb":
        detector = cv2.ORB_create(nfeatures=num_features)
    elif detector == "kaze":
        detector = cv2.KAZE_create()
    elif detector == "akaze":
        detector = cv2.AKAZE_create()
    elif detector == "brisk":
        detector = cv2.BRISK_create()
    else:
        raise Exception("Unsupported detector.")

    if use_latch_descriptor:

        # Instantiate LATCH descriptor
        latch = cv2.xfeatures2d.LATCH_create(64, True, 15)

        # Detect
        kp1 = detector.detect(source)
        kp2 = detector.detect(target)

        # Compute
        _, des1 = latch.compute(source, kp1)
        _, des2 = latch.compute(target, kp2)

    else:

        kp1, des1 = detector.detectAndCompute(source, None)
        kp2, des2 = detector.detectAndCompute(target, None)

    # Create control image if needed
    source_descr = None
    target_descr = None
    if control_image:
        source_descr = cv2.drawKeypoints(source, kp1, source, color=(0, 255, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        target_descr = cv2.drawKeypoints(target, kp2, target, color=(0, 255, 0),
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2SQR, crossCheck=True)
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort matches by score (distance)
    matches = sorted(matches, key=lambda x: x.distance)

    view = None
    if control_image:
        # Create quality control image
        view = display_matches(source, target, matches, kp1, kp2, 20)

    # Get and cast the points to be used for transformation estimation
    s_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    d_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    results = {
        "perspective": None,
        "affine": None,
        "rigid": None
    }

    if s_pts.size == 0 or d_pts.size == 0:
        # No matches found! 
        return results, None, None, None

    # Estimate the image transformation
    if rigid:

        M, mask = cv2.estimateAffinePartial2D(s_pts, d_pts, cv2.RANSAC)

        # Transform source based on the extracted transformation
        if M is not None:
            aligned = cv2.warpAffine(source, M, source.shape[0:2])
        else:
            aligned = source.copy()

        # Add the results dictionary
        results["rigid"] = aligned, M, mask

    if affine:

        M, mask = cv2.estimateAffine2D(s_pts, d_pts, cv2.RANSAC)

        # Transform source based on the extracted transformation
        if M is not None:
            aligned = cv2.warpAffine(source, M, source.shape[0:2])
        else:
            aligned = source.copy()

        # Add the results dictionary
        results["affine"] = aligned, M, mask

    if perspective:

        M, mask = cv2.findHomography(s_pts, d_pts, cv2.RANSAC, 5.0)

        # Warp source based on the extracted transformation
        if M is not None:
            aligned = cv2.warpPerspective(source, M, source.shape[0:2])
        else:
            aligned = source.copy()

        # Add the results dictionary
        results["perspective"] = aligned, M, mask


    # Return
    return results, view, source_descr, target_descr


def apply_transformation_to_image(image, transformation_type, transformation_matrix, target_height=None, target_width=None):
    """Apply a transformation to an image.

    Keyword arguments:
    :param image: image to be transformed.
    :param transformation_type: type of transformation. One of:
        "perspective": register the image using a perspective transformation.
        "affine": register the image using an affine transformation.
        "rigid": register the image using a rigid transformation.
    :param transformation_matrix: transformation matrix, must be:
        "perspective": (3x3)
        "affine": (2x3)
        "rigid": (2x3)
    :param target_height: (optional) number of rows of the transformed image. If not set, the transformed image
        will have the same size as the source image.
    :param target_width: (optional) number of columns of the transformed image. If not set, the transformed image
        will have the same size as the source image.
    :return: transformed: transformed image.
    """

    if target_height is None or target_width is None:
        target_size = (image.shape[1], image.shape[0])
    else:
        target_size = (target_height, target_width)

    # Apply the image transformation
    if transformation_type == "rigid":
        transformed = cv2.warpAffine(image, transformation_matrix, target_size)

    elif transformation_type == "affine":       
        transformed = cv2.warpAffine(image, transformation_matrix, target_size)

    elif transformation_type == "perspective":
        transformed = cv2.warpPerspective(image, transformation_matrix, target_size)
    else:
        raise Exception(f"Invalid transformation type {transformation_type}!")

    # Return
    return transformed


def display_matches(img1, img2, sel_matches, k1, k2, max_matches=None):
    """Displays the matches on a control image and returns it."""

    # If img1 and img2 are RGB, convert them to gray values first
    if np.ndim(img1) == 3:
        img1 = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
    if np.ndim(img2) == 3:
        img2 = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    view[:h1, :w1, 0] = img1
    view[:h2, w1:, 0] = img2
    view[:, :, 1] = view[:, :, 0]
    view[:, :, 2] = view[:, :, 0]

    n = 0
    for m in sel_matches:
        # Draw the key points
        color = tuple([np.random.randint(0, 255) for _ in range(3)])
        try:
            cv2.line(view,
                     (int(k1[m.queryIdx].pt[0]),
                      int(k1[m.queryIdx].pt[1])),
                     (int(k2[m.trainIdx].pt[0] + w1),
                      int(k2[m.trainIdx].pt[1])),
                     color)
            n += 1
            if max_matches is not None:
                if n >= max_matches:
                    break
        except:
            print(m.queryIdx, m.trainIdx)

    return view


def add_border(images: list, border: int, fill_value: int = -1) -> list:
    """Add a border to each of the images in a list and sets the border values to a given fill value.

    If the fill_value is omitted, the median of all pixel intensities will taken.
    """
    out = []
    for img in images:
        if fill_value == -1:
            curr_fill_value = np.median(img)
        else:
            curr_fill_value = fill_value
        if img.ndim == 2:
            img_ext = np.full((img.shape[0] + 2 * border, img.shape[1] + 2 * border), curr_fill_value, dtype=img.dtype)
            img_ext[border:-border, border:-border] = img
        elif img.ndim == 3:
            img_ext = np.full((img.shape[0] + 2 * border, img.shape[1] + 2 * border, 3), curr_fill_value, dtype=img.dtype)
            for i in range(3):
                img_ext[border:-border, border:-border, i] = img[:, :, i]
        else:
            raise Exception(f"Unexpected number of dimensions ({img.ndim})!")
        out.append(img_ext)

    return out


def BGR2Gray(image, to_lightness=False):
    """Convert a BGR image to gray or lightness."""

    # If the image is already 2D, do nothing
    if image.ndim == 2:
        return image

    if to_lightness:

        # Convert to lab
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, _, _ = cv2.split(lab)

        # Return lightness
        return l

    else:

        # Convert to gray and return
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
