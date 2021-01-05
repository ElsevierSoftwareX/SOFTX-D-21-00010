import os
import platform
import site
import sys
import shutil
from pathlib import Path
import json


# Try adding the python devel libs to LD_LIBRARY_PATH
if sys.platform == "linux":

    # Python library path
    python_dir = site.getsitepackages()
    packages_root_dir = python_dir[0]
    python_library_path = str(Path(packages_root_dir, '../..').resolve())

    # Keep track of possible interpreter restart
    restart_required = False

    # If necessary, add to the LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' not in os.environ:

        # LD_LIBRARY_PATH is not set. Add python_library_path
        os.environ['LD_LIBRARY_PATH'] = python_library_path

        # Flag for restart
        restart_required = True

    else:

        # Get current list
        paths = os.environ['LD_LIBRARY_PATH']
        path_list = paths.split(':')

        # Is the python library path in the list?
        if python_library_path not in path_list:

            # Append python_library_path
            paths = paths + ':' + python_library_path

            # Update LD_LIBRARY_PATH in the environment
            os.environ['LD_LIBRARY_PATH'] = paths

            # Flag for restart
            restart_required = True

    if restart_required:

        # Inform
        print(f"Adding {python_library_path} to "
              f"LD_LIBRARY_PATH and restarting the interpreter.")

        try:
            os.execv(sys.executable, ['python'] + sys.argv)
        except Exception as exc:
            print(f"Failed re-exec: {exc}")
            sys.exit(1)

try:
    from pypocquant.manual import build_manual
except:
    pypocquant_path = Path(__file__).resolve().parent / 'src' / 'main' / 'python' / 'pyPOCQuantUI'
    sys.path.insert(0, str(pypocquant_path))

from pypocquant.manual import build_manual, build_quickstart
from ui import versionInfo


def update_application_metadata():
    """Update application metadata before freeze."""

    # Get the path to src/build/settings/base.json
    current_folder = Path(__file__).parent.resolve()
    base_json_path = current_folder / "src" / "build" / "settings" / "base.json"

    # Make sure that the file exists
    if not base_json_path.is_file():
        print(f"Settings file {str(base_json_path)} not found! Cannot update metadata")
        return

    # Read the file
    with open(base_json_path) as json_file:
        metadata = json.load(json_file)

    # Update the version
    metadata["version"] = versionInfo.get_version_string()

    # Save the file
    with open(base_json_path, 'w') as json_file:
        json.dump(metadata, json_file)


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

    # Build the list of files and folder to copy to the correct `freeze` subfolder
    if platform.system() == 'Windows':

        packages_root_dir = python_dir[1]

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
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/windows/ui')
        )
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/windows')
        )

    elif platform.system() == 'Darwin':

        packages_root_dir = python_dir[0]

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
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/mac/Contents/MacOS/ui')
        )
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/mac/Contents/MacOS')
        )

    elif platform.system() == 'Linux':

        packages_root_dir = python_dir[0]

        print(f"Packages root dir = {packages_root_dir}")

        # Build map of files/folders to copy
        item_list = [
            'cairosvg/VERSION',
            'cairocffi/VERSION',
            'sklearn/utils/_weight_vector.cpython-36m-x86_64-linux-gnu.so',
            'scipy/special/cython_special.cpython-36m-x86_64-linux-gnu.so',
            'pywt/_extensions/_cwt.cpython-36m-x86_64-linux-gnu.so',
            'skimage/feature/orb_cy.cpython-36m-x86_64-linux-gnu.so',
            'skimage/feature/_orb_descriptor_positions.py',
            'sklearn/utils/_cython_blas.cpython-36m-x86_64-linux-gnu.so'
        ]

        # Copy missing files
        copy_and_print_missing_files(
            packages_root_dir,
            Path(target_root_dir / 'src/freeze/linux'),
            item_list
        )

        # Copy version info
        Path(target_root_dir / 'src/freeze/linux/ui').mkdir(parents=True, exist_ok=True)
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/linux/ui')
        )
        shutil.copy(
            Path(target_root_dir / 'src/main/python/pyPOCQuantUI/ui/VERSION'),
            Path(target_root_dir / 'src/freeze/linux')
        )

    else:

        raise Exception("Platform not supported!")


print('|---------------------------------------------------------------------------------------------------|')
print('| Start building pyPOCQUANT')
print('|---------------------------------------------------------------------------------------------------|')

# First, update application version
update_application_metadata()

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
