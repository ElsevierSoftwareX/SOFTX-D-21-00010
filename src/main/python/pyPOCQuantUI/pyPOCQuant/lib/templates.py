from pathlib import Path
import cv2

from pyPOCQuant.lib.processing import BGR2Gray
from pyPOCQuant.lib.utils import get_templates_folder


def load_templates_orig(scaled_size=(237, 837)):
    """Load templates to be used for registrations.

    :param scaled_size: tuple
        Final dimension of the template: (height, width)
    """

    # Read from the pyPOCQuant template folder
    template_folder = get_templates_folder()

    # Add templates to a dictionary
    templates = {
        "template_right": cv2.imread(str(Path(template_folder) / "template_strip_right.png")),
        "template_left": cv2.imread(str(Path(template_folder) / "template_strip_left.png")),
        "template_alt_right": cv2.imread(str(Path(template_folder) / "template_strip_alt_right.png")),
        "template_alt_left": cv2.imread(str(Path(template_folder) / "template_strip_alt_left.png")),
        "template_barcode_right": cv2.imread(str(Path(template_folder) / "template_barcode_right.png")),
        "template_barcode_left": cv2.imread(str(Path(template_folder) / "template_barcode_left.png")),
        "template_barcode_up": cv2.imread(str(Path(template_folder) / "template_barcode_up.png")),
        "template_barcode_down": cv2.imread(str(Path(template_folder) / "template_barcode_down.png"))
    }

    # Do we need to scale the barcode images? We can check on one template only,
    # since they all have the same dimensions. We do not assume that template and
    # barcode will be oriented in the same direction in the image, so we pick the
    # axis with the largest scaling factor with respect to 'scaled_size' and use
    # that for an isotropic scaling of t he barcode templates. Here there is an
    # assumption: the strip might me larger or smaller than the template, but the
    # aspect ratio should remain more or less the same.
    scaling_factor = 1.0
    if templates["template_right"].shape[:2] != scaled_size:

        # Which is the most important axis?
        scaling_factor_x = float(scaled_size[1]) / float(templates["template_right"].shape[1])
        if scaled_size[1] >= templates["template_right"].shape[1]:
            cmp_scaling_factor_x = scaling_factor_x
        else:
            cmp_scaling_factor_x = 1.0 / scaling_factor_x

        scaling_factor_y = float(scaled_size[0]) / float(templates["template_right"].shape[0])
        if scaled_size[0] >= templates["template_right"].shape[0]:
            cmp_scaling_factor_y = scaling_factor_y
        else:
            cmp_scaling_factor_y = 1.0 / scaling_factor_y

        if cmp_scaling_factor_x >= cmp_scaling_factor_y:
            scaling_factor = scaling_factor_x
        else:
            scaling_factor = scaling_factor_y

    # Scale the templates if necessary
    if templates["template_right"].shape[:2] != scaled_size:
        templates["template_right"] = cv2.resize(
            templates["template_right"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    if templates["template_left"].shape[:2] != scaled_size:
        templates["template_left"] = cv2.resize(
            templates["template_left"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    if templates["template_alt_right"].shape[:2] != scaled_size:
        templates["template_alt_right"] = cv2.resize(
            templates["template_alt_right"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    if templates["template_alt_left"].shape[:2] != scaled_size:
        templates["template_alt_left"] = cv2.resize(
            templates["template_alt_left"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    # Scale the barcode templates if necessary
    if scaling_factor != 1.0:
        barcode_scaled_size = (
            int(templates["template_barcode_right"].shape[1] * scaling_factor),
            int(templates["template_barcode_right"].shape[0] * scaling_factor),
        )
        templates["template_barcode_right"] = cv2.resize(
            templates["template_barcode_right"], dsize=barcode_scaled_size, interpolation=cv2.INTER_LANCZOS4
        )

        barcode_scaled_size = (
            int(templates["template_barcode_left"].shape[1] * scaling_factor),
            int(templates["template_barcode_left"].shape[0] * scaling_factor),
        )
        templates["template_barcode_left"] = cv2.resize(
            templates["template_barcode_left"], dsize=barcode_scaled_size, interpolation=cv2.INTER_LANCZOS4
        )

        barcode_scaled_size = (
            int(templates["template_barcode_up"].shape[1] * scaling_factor),
            int(templates["template_barcode_up"].shape[0] * scaling_factor),
        )
        templates["template_barcode_up"] = cv2.resize(
            templates["template_barcode_up"], dsize=barcode_scaled_size, interpolation=cv2.INTER_LANCZOS4
        )

        barcode_scaled_size = (
            int(templates["template_barcode_down"].shape[1] * scaling_factor),
            int(templates["template_barcode_down"].shape[0] * scaling_factor),
        )
        templates["template_barcode_down"] = cv2.resize(
            templates["template_barcode_down"], dsize=barcode_scaled_size, interpolation=cv2.INTER_LANCZOS4
        )

    # Add gray-value version of all templates
    templates = {
        "template_right_gray": BGR2Gray(templates["template_right"]),
        "template_left_gray": BGR2Gray(templates["template_left"]),
        "template_alt_right_gray": BGR2Gray(templates["template_alt_right"]),
        "template_alt_left_gray": BGR2Gray(templates["template_alt_left"]),
        "template_barcode_right_gray": BGR2Gray(templates["template_barcode_right"]),
        "template_barcode_left_gray": BGR2Gray(templates["template_barcode_left"]),
        "template_barcode_up_gray": BGR2Gray(templates["template_barcode_up"]),
        "template_barcode_down_gray": BGR2Gray(templates["template_barcode_down"])
    }

    return templates


def load_templates_TPH(scaled_size=(237, 870)):
    """Load templates to be used for registrations.

    :param scaled_size: tuple
        Final dimension of the template: (height, width)
    """

    # Read from the pyPOCQuant template folder
    template_folder = get_templates_folder()

    # Add templates to a dictionary
    templates = {
        "template_right": cv2.imread(str(Path(template_folder) / "template_strip_cropped_right.png")),
        "template_left": cv2.imread(str(Path(template_folder) / "template_strip_cropped_left.png")),
    }

    # Scale if necessary
    if templates["template_right"].shape[:2] != scaled_size:
        templates["template_right"] = cv2.resize(
            templates["template_right"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    if templates["template_left"].shape[:2] != scaled_size:
        templates["template_left"] = cv2.resize(
            templates["template_left"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    # Add gray-value version of all templates
    templates = {
        "template_right_gray": BGR2Gray(templates["template_right"]),
        "template_left_gray": BGR2Gray(templates["template_left"]),
    }

    return templates


def load_templates_FH(scaled_size=(237, 870)):
    """Load templates to be used for registrations.

    :param scaled_size: tuple
        Final dimension of the template: (height, width)
    """

    # Read from the pyPOCQuant template folder
    template_folder = get_templates_folder()

    # Add templates to a dictionary
    templates = {
        "template_right": cv2.imread(str(Path(template_folder) / "template_strip_fh_white_right.png")),
        "template_left": cv2.imread(str(Path(template_folder) / "template_strip_fh_white_left.png")),
    }

    # Scale if necessary
    if templates["template_right"].shape[:2] != scaled_size:
        templates["template_right"] = cv2.resize(
            templates["template_right"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    if templates["template_left"].shape[:2] != scaled_size:
        templates["template_left"] = cv2.resize(
            templates["template_left"], dsize=(scaled_size[1], scaled_size[0]), interpolation=cv2.INTER_LANCZOS4
        )

    # Add gray-value version of all templates
    templates = {
        "template_right_gray": BGR2Gray(templates["template_right"]),
        "template_left_gray": BGR2Gray(templates["template_left"]),
    }

    return templates
