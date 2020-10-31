import os
import platform
import shutil
import sys
from pathlib import Path
import site

from pypocquant.manual import build_manual, build_quickstart
from ui import versionInfo


def copy_and_print_missing_files(source_root_dir: Path, target_root_dir: Path, list_items: list):
    """Copy missing files before freezing.

    """

    # Make sure that both source_root_dir and target_root_dir are Path onjects
    source_root_dir = Path(source_root_dir)
    target_root_dir = Path(target_root_dir)

    # Clean the target
    if target_root_dir.is_dir():
        shutil.rmtree(target_root_dir)
    target_root_dir.mkdir(parents=True, exist_ok=True)

    # Now copy the items from the list
    for item in list_items:
        source = Path(source_root_dir / item)
        target = Path(target_root_dir / item)
        if source.is_dir():
            shutil.copytree(source, target)
            item_type = "DIR"
        elif source.is_file():
            parent_folder = Path(target.parents[0].resolve())
            parent_folder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            item_type = "FILE"
        else:
            print(f"[ERROR] '{source}' not found.")
            continue

        print(f"[{item_type:5s}] Copied from '{source}' to '{target}'.")


def prepare_freezing():
    """Copy files known to be missed by `fbs freeze`."""

    # Target root folder
    target_root_dir = Path(Path(__file__).parents[0].resolve())

    # Create common folders
    Path(target_root_dir / 'src/freeze').mkdir(parents=True, exist_ok=True)
    Path(target_root_dir / 'src/freeze/windows').mkdir(parents=True, exist_ok=True)
    Path(target_root_dir / 'src/freeze/osx').mkdir(parents=True, exist_ok=True)
    Path(target_root_dir / 'src/freeze/linux').mkdir(parents=True, exist_ok=True)

    # Source root folder
    python_dir = site.getsitepackages()
    packages_root_dir = python_dir[1]

    # Build the list of files and folder to copy to the correct `freeze` subfolder
    if platform.system() == 'Windows':

        # Build map of files/folders to copy
        item_list = [
            'cairosvg',
            'cssselect2/VERSION',
            'tinycss2/VERSION',
            'sklearn/utils/_cython_blas.cp36-win_amd64.pyd',
            'skimage/feature/_orb_descriptor_positions.py',
            'dask/dask.yaml',
            'pywt/_extensions/_cwt.cp36-win_amd64.pyd',
            'pyzbar',
            'scipy/.libs',
            'scipy/special/cython_special.cp36-win_amd64.pyd'
        ]

        # Copy missing files
        copy_and_print_missing_files(
            packages_root_dir,
            Path(target_root_dir / 'src/freeze/windows'),
            item_list
        )
        
        # Copy version info
        Path(target_root_dir / 'src/freeze/windows/ui').mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'), Path(target_root_dir / 'src/freeze/windows/ui'))
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'), Path(target_root_dir / 'src/freeze/windows'))
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/BUILD'), Path(target_root_dir / 'src/freeze/windows/ui'))


        # Original list -- kept for reference
        # print(str(Path(packages_root_dir, 'cairosvg')), str(Path('src/freeze/windows/cairosvg')))
        # print(str(Path(packages_root_dir,'cssselect2','VERSION')), str(Path('src/freeze/windows/cssselect2')))
        # print(str(Path(packages_root_dir, 'tinycss2', 'VERSION')), str(Path('src/freeze/windows/tinycss2')))
        # print(str(Path(packages_root_dir, 'sklearn/utils', '_cython_blas.cp36-win_amd64')),
        #       str(Path('src/freeze/windows/sklearn/utils')))
        # print(str(Path(packages_root_dir, 'skimage/feature', '_orb_descriptor_positions')),
        #       str(Path('src/freeze/windows/skimage/feature')))
        # print(str(Path(packages_root_dir, 'dask', 'dask.yaml')), str(Path('src/freeze/windows/dask')))
        # print(str(Path(packages_root_dir, 'pywt/_extensions', '_cwt.cp36-win_amd64')),
        #       str(Path('src/freeze/windows/pywt/_extensions')))
        # print(str(Path(packages_root_dir, 'pyzbar')), str(Path('src/freeze/windows/pyzbar')))
        # print(str(Path(packages_root_dir, 'scipy/.libs', )), str(Path('src/freeze/windows/scipy/')))
        # print(str(Path(packages_root_dir, 'scipy/special', 'cython_special.cp36-win_amd64')),
        #       str(Path('src/freeze/windows/scipy/special')))
    elif platform.system() == 'Darwin':

        # Build map of files/folders to copy
        item_list = [
            'cairosvg/VERSION',
            'cairocffi/VERSION',
            'cv2',
            'reportlab',
            'skimage',
            'sklearn',
            'pywt',
        ]

        # Copy missing files
        copy_and_print_missing_files(
            packages_root_dir,
            Path(target_root_dir / 'src/freeze/mac/Contents/MacOS'),
            item_list
        )

        # Copy version info
        Path(target_root_dir / 'src/freeze/mac/Contents/MacOS').mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
                    Path(target_root_dir / 'src/freeze/mac/Contents/MacOS/ui'))
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
                    Path(target_root_dir / 'src/freeze/mac/Contents/MacOS'))
        shutil.copy(Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/BUILD'),
                    Path(target_root_dir / 'src/freeze/mac/Contents/MacOS/ui'))


print('|---------------------------------------------------------------------------------------------------|')
print('| Start building pyPOCQUANT')
print('|---------------------------------------------------------------------------------------------------|')

__VERSION__ = "0.9.0"

# Update version and build info
if not versionInfo.compare_version(__VERSION__):
    versionInfo.set_new_version(__VERSION__)
versionInfo.increase_build_number()

# Build manual
build_manual()
# Build quickstart
build_quickstart()

# clean the app out dir
print('Running: fbs clean')
stream = os.popen('fbs clean')
print(stream.read())

# print('Running: fbs run')
# print('Note: test if app works. Close it to continue')
# stream = os.popen('fbs run')
# print(stream.read())

print('Prepare freezing')
prepare_freezing()

# print('Running: fbs freeze')
stream = os.popen('fbs freeze')
print(stream.read())
#
# print('Running: fbs installer')
stream = os.popen('fbs installer')
print(stream.read())


print('|---------------------------------------------------------------------------------------------------|')
print('| Done.')
print('|---------------------------------------------------------------------------------------------------|')

sys.exit(0)
