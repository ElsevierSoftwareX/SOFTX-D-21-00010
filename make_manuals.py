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

import sys
from pathlib import Path

try:
    from pypocquant.manual import build_manual
except:
    pypocquant_path = Path(__file__).resolve().parent / 'src' / 'main' / 'python' / 'pyPOCQuantUI'
    sys.path.insert(0, str(pypocquant_path))

from pypocquant.manual import build_manual, build_quickstart

print('|---------------------------------------------------------------------------------------------------|')
print('| Start building manual')
print('|---------------------------------------------------------------------------------------------------|')

# Build manual
build_manual()

# Build quickstart
build_quickstart()

print('|---------------------------------------------------------------------------------------------------|')
print('| Done.')
print('|---------------------------------------------------------------------------------------------------|')

sys.exit(0)
