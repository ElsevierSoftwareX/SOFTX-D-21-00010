from enum import Enum
from pyzbar.pyzbar import ZBarSymbol


class Issue(Enum):
    NONE = 0
    BARCODE_EXTRACTION_FAILED = 1
    FID_EXTRACTION_FAILED = 2
    POOR_STRIP_ALIGNMENT = 3
    SENSOR_EXTRACTION_FAILED = 4
    BAND_QUANTIFICATION_FAILED = 5
    CONTROL_BAND_MISSING = 6
    STRIP_BOX_EXTRACTION_FAILED = 7


class SymbolTypes(Enum):
    TYPES = [ZBarSymbol.CODE39, ZBarSymbol.CODE128, ZBarSymbol.QRCODE]


# List of known strip manufacturers
KnownManufacturers = (
    'AUGURIX',
    'BIOZAK',
    'CTKBIOTECH',
    'DRALBERMEXACARE',
    'LUMIRATEK',
    'NTBIO',
    'SUREBIOTECH',
    'TAMIRNA'
)
