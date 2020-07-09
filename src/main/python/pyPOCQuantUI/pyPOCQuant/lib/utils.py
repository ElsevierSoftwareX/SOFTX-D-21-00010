import os
import getpass
import rawpy
import imageio
from pathlib import Path
from sys import platform
import pytesseract
import exifread
from datetime import datetime
import pandas as pd
import sys
import time
import cv2


def create_quality_control_images(results_folder_path: str, basename: str, map_of_images: dict):
    """Save the list of requested quality control images."""

    for key in map_of_images:
        if map_of_images[key] is None:
            continue
        cv2.imwrite(str(results_folder_path / Path(basename + "_" + key + ".png")), map_of_images[key])

def get_project_root() -> Path:
    """Returns project root folder."""
    try:
        # Running from a pyinstaller executable
        project_root = Path(sys._MEIPASS)
    except:
        project_root = Path(__file__).parent.parent
    return project_root


def get_templates_folder() -> Path:
    """Returns templates folder."""
    try:
        # Running from a pyinstaller executable
        templates_folder = Path(sys._MEIPASS) / "templates"
    except:
        templates_folder = get_project_root() / "pyPOCQuant" / "templates"
    return templates_folder


def get_data_folder() -> Path:
    """Returns the value of the environment variable DATA_FOLDER or,
    if not found, the value if `get_project_root()`."""

    data_folder = ""
    if "DATA_FOLDER" in os.environ:
        data_folder = os.environ['DATA_FOLDER']

    if data_folder == "":
        data_folder = get_project_root()
    else:
        data_folder = Path(data_folder)
    return data_folder


def image_format_converter(directory, filename, output_dir=None, image_format='tif'):
    """Converts a image in raw format (.'nef') to the specified open format. Default is '.tif'.
       rawpy API: https://letmaik.github.io/rawpy/api/rawpy.RawPy.html,
                  https://letmaik.github.io/rawpy/api/rawpy.Params.html

    """

    with rawpy.imread(str(directory.joinpath(filename))) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=False, output_bps=16)

        basename = Path(filename).stem
        if output_dir:
            imageio.imsave(str(output_dir.joinpath('{}.{}'.format(basename, image_format))), rgb)
        else:
            imageio.imsave(str(directory.joinpath('{}.{}'.format(basename, image_format))), rgb)


def get_iso_date_from_image(image_path):
    # get all Exif image metadata
    f = open(image_path, 'rb')
    tags = exifread.process_file(f)
    f.close()
    try:
        # Convert datetime string to iso date
        date = datetime.strptime(tags['EXIF DateTimeOriginal'].values,
                                 '%Y:%m:%d %H:%M:%S')
        iso_date = date.strftime('%Y-%m-%d')
        iso_time = date.strftime('%H-%M-%S')
    except Exception as e:
        try:
            date = datetime.strptime(time.ctime(os.path.getctime(image_path)), "%a %b %d %H:%M:%S %Y")
            iso_date = date.strftime('%Y-%m-%d')
            iso_time = date.strftime('%H-%M-%S')
        except Exception as e:
            iso_date = -1
            iso_time = -1

    return iso_date, iso_time


def get_exif_details(image_path):
    # get all Exif image metadata
    f = open(image_path, 'rb')
    tags = exifread.process_file(f)
    f.close()

    try:
        exp_time = tags['EXIF ExposureTime'].values
    except Exception as e:
        exp_time = -1

    try:
        f_number = tags['EXIF FNumber'].values
    except Exception as e:
        f_number = -1

    try:
        focal_length_35_mm = tags['EXIF FocalLengthIn35mmFilm'].values
    except Exception as e:
        focal_length_35_mm = -1

    try:
        iso_speed = tags['EXIF ISOSpeedRatings'].values
    except Exception as e:
        try:
            iso_speed = tags['MakerNote ISOSpeedRequested'].values[1]
        except Exception as e:
            iso_speed = -1

    return exp_time, f_number, focal_length_35_mm, iso_speed

def get_orientation_from_image(image_path):
    # get all Exif image metadata
    f = open(image_path, 'rb')
    tags = exifread.process_file(f)
    # Get the orientation
    orientation = str(tags['Image Orientation'].printable)
    f.close()
    return orientation


def is_on_path(prog):
    for root_dir in os.environ['PATH'].split(os.pathsep):
        if os.path.exists(os.path.join(root_dir, prog)):
            return True
    return False


def set_tesseract_exe():
    # @todo improve.
    if is_on_path('Tesseract-OCR'):
        return
    else:
        # Check default installations
        if platform == "linux" or platform == "linux2":
            pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
        elif platform == "darwin":
            pytesseract.pytesseract.tesseract_cmd = r'/Users/{}/tesseract-ocr/tesseract'.format(getpass.getuser())
        elif platform == "win32":
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def remove_filename_duplicates(data_frame):
    df = data_frame.copy()
    dff = pd.Series([False] * df.shape[0])
    filename_no_ext = [os.path.splitext(x)[0] for x in df.FILENAME]
    filename_ext = [os.path.splitext(x)[1] for x in df.FILENAME]
    df['NAME'] = filename_no_ext
    df['NAMEEXT'] = filename_ext
    unique_names = df.NAME.unique()

    for uname in unique_names:
        sel = df[df['NAME'] == uname]

        if sel.shape[0] > 1:
            idx = sel.NAMEEXT.str.lower() == '.nef'
            if sel[idx].FID.values == sel[~idx].FID.values:
                # save index of nef image
                dff[df['NAME'] == uname] = idx
            elif sel[idx].FID.values == '':
                # save index of jpg / jpeg or any other filename ending since nef file has no fid
                dff[df['NAME'] == uname] = ~idx
            else:
                # @todo review case
                # non empty nef file but non matching with jpg. what to do? lets trust raw more and keep raw fid
                dff[df['NAME'] == uname] = idx
        else:
            # If filename is NO duplicate keep it if none empty otherwise drop
            if sel.FID.values != '':
                dff[df['NAME'] == uname] = True
            else:
                dff[df['NAME'] == uname] = False

    return data_frame[dff]
