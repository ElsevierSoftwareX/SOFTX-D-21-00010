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

import unittest
from pypocquant import tests

# Run all unit tests
suite = unittest.TestLoader().loadTestsFromModule(tests)
result = unittest.TextTestRunner().run(suite)

print(f"Run {result.testsRun} tests: "
      f"{len(result.errors)} error(s) and "
      f"{len(result.failures)} failure(s).")
