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
from pathlib import Path

def run_process(command):
    working_dir = str(Path(__file__).parent / "docs")
    p = subprocess.Popen(command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         cwd=working_dir)
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

        manual_dir = Path(__file__).parent / "src" / "main" / "python" / "pyPOCQuantUI" / "pypocquant" / "manual"
        docs_dir = Path(__file__).parent / "docs"
        src = [
            str(manual_dir / "ui_images"),
            str(manual_dir / "demo_image"),
            str(manual_dir / "problem_solutions"),
            str(manual_dir / "setup"),
            str(manual_dir / "QuickStart.md"),
            str(manual_dir / "UserInstructions.md")
        ]
        dst = [
            str(docs_dir / "ui_images"),
            str(docs_dir / "demo_image"),
            str(docs_dir / "problem_solutions"),
            str(docs_dir / "setup"),
            str(docs_dir / "QuickStart.md"),
            str(docs_dir / "UserInstructions.md")
        ]

        for idx in range(0, len(src)):
            copy_files(src[idx], dst[idx])

        print('building docs')

        # new build
        run_process('make html')

        # copy missing resources to build directory
        print('copy files')
        src = [
            str(docs_dir / "ui_images"),
            str(docs_dir / "demo_image"),
            str(docs_dir / "problem_solutions"),
            str(docs_dir / "setup")
        ]
        dst = [
            str(docs_dir / "_build" / "html" / "ui_images"),
            str(docs_dir / "_build" / "html" / "demo_image"),
            str(docs_dir / "_build" / "html" / "problem_solutions"),
            str(docs_dir / "_build" / "html" / "setup")
        ]

        for idx in range(0, len(src)):
            copy_files(src[idx], dst[idx])

        print('done')

    except Exception as error:
        print(error)


if __name__ == '__main__':
    make_doc()
