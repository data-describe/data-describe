from enum import Enum


class _FileExtensionTypes(Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"


_FILE_EXTENSION_MAPPING = {
    "csv": [".csv"],
    "json": [".json"],
    "excel": [".xls", ".xlsx", ".xlsb", ".xlsm"],
}


def is_filetype(filetype: str, extension: str) -> bool:
    """Checks if the file extension matches a given file type.

    Args:
        filetype (str): The type of file. See the enum _FileExtensionTypes
        extension (str): The file extension

    Returns:
        bool: True if matching
    """
    return extension in _FILE_EXTENSION_MAPPING.get(filetype, [])
