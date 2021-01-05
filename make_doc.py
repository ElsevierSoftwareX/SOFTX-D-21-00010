#  ********************************************************************************
#   Copyright © 2020-2021, ETH Zurich, D-BSSE, Andreas P. Cuny & Aaron Ponti
#   All rights reserved. This program and the accompanying materials
#   are made available under the terms of the GNU Public License v3.0
#   which accompanies this distribution, and is available at
#   http://www.gnu.org/licenses/gpl
#
#   Contributors:
#     * Andreas P. Cuny - initial API and implementation
#     * Aaron Ponti - initial API and implementation
#  *******************************************************************************

import os
import sys
import errno
import shutil
import subprocess


def run_process(command):
    p = subprocess.Popen(command,
                         shell=False,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd='docs')
    stdout, stderr = p.communicate()


def copy_files(src, dst):
    try:
        if os.path.exists(dst):
            pass
        else:
            shutil.copytree(src, dst)
    except OSError as e:
        if e.errno == errno.ENOTDIR:
            os.remove(dst)
            shutil.copy(src, dst)
        else:
            print('Directory not copied. Error: %s' % e)


def make_doc():
    """Run Sphinx to build the doc."""
    try:
        # removing previous build
        print('BUILDING DOCS')
        print('=============')
        print('removing previous build')

        run_process('make clean')

        print('copy files')

        src = [r'src\main\python\pyPOCQuantUI\pypocquant\manual\ui_images',
               r'src\main\python\pyPOCQuantUI\pypocquant\manual\demo_image',
               r'src\main\python\pyPOCQuantUI\pypocquant\manual\problem_solutions',
               r'src\main\python\pyPOCQuantUI\pypocquant\manual\setup',
               r'src\main\python\pyPOCQuantUI\pypocquant\manual\QuickStart.md',
               r'src\main\python\pyPOCQuantUI\pypocquant\manual\UserInstructions.md', ]
        dst = [r'docs\ui_images', r'docs\demo_image', r'docs\problem_solutions',
               r'docs\setup', r'docs\QuickStart.md', r'docs\UserInstructions.md']

        for idx in range(0, len(src)):
            copy_files(src[idx], dst[idx])

        print('building docs')

        # new build
        run_process('make html')

        # copy missing resources to build directory
        print('copy files')
        src = [r'docs\ui_images', r'docs\demo_image', r'docs\problem_solutions', r'docs\setup']
        dst = [r'docs\_build\html\ui_images', r'docs\_build\html\demo_image',
               r'docs\_build\html\problem_solutions', r'docs\_build\html\setup']

        for idx in range(0, len(src)):
            copy_files(src[idx], dst[idx])

        print('done')

    except Exception as error:
        print(error)


if __name__ == '__main__':
    make_doc()
