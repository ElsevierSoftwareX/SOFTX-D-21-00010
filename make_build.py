import os
import platform
import shutil
from pathlib import Path
import site


# @todo add versioning

def prepare_freezing():
    if platform.system() == 'Windows':
        # Create relevant folders
        Path('src/freeze').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/osx').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/cairosvg').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/cssselect2').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/dask').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/pywt').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/pywt/_extensions').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/pyzbar').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/scipy').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/scipy/special').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/skimage').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/skimage/feature').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/sklearn').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/sklearn/utils').mkdir(parents=True, exist_ok=True)
        Path('src/freeze/windows/tinycss2').mkdir(parents=True, exist_ok=True)

        python_dir = site.getsitepackages()
        packages_root_dir = python_dir[1]

        # Copy missing files
        # @todo replace print with shutil.copy2 and keep print to inform user
        print(str(Path(packages_root_dir, 'cairosvg')), str(Path('src/freeze/windows/cairosvg')))
        print(str(Path(packages_root_dir,'cssselect2','VERSION')), str(Path('src/freeze/windows/cssselect2')))
        print(str(Path(packages_root_dir, 'tinycss2', 'VERSION')), str(Path('src/freeze/windows/tinycss2')))
        print(str(Path(packages_root_dir, 'sklearn/utils', '_cython_blas.cp36-win_amd64')),
              str(Path('src/freeze/windows/sklearn/utils')))
        print(str(Path(packages_root_dir, 'skimage/feature', '_orb_descriptor_positions')),
              str(Path('src/freeze/windows/skimage/feature')))
        print(str(Path(packages_root_dir, 'dask', 'dask.yaml')), str(Path('src/freeze/windows/dask')))
        print(str(Path(packages_root_dir, 'pywt/_extensions', '_cwt.cp36-win_amd64')),
              str(Path('src/freeze/windows/pywt/_extensions')))
        print(str(Path(packages_root_dir, 'pyzbar')), str(Path('src/freeze/windows/pyzbar')))
        print(str(Path(packages_root_dir, 'scipy/.libs', )), str(Path('src/freeze/windows/scipy/')))
        print(str(Path(packages_root_dir, 'scipy/special', 'cython_special.cp36-win_amd64')),
              str(Path('src/freeze/windows/scipy/special')))


print('|-------------------------------------------------------------------------------------------------------------|')
print('| Start building pyPOCQUANT')
print('|-------------------------------------------------------------------------------------------------------------|')

# @todo fix paths
# exec(open(r'main\python\pyPOCQuantUI\pypocquant\manual\make_manual_release.py').read())

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
# stream = os.popen('fbs freeze')
# print(stream.read())
#
# print('Running: fbs installer')
# stream = os.popen('fbs installer')
# print(stream.read())


print('|-------------------------------------------------------------------------------------------------------------|')
print('| Done.')
print('|-------------------------------------------------------------------------------------------------------------|')




