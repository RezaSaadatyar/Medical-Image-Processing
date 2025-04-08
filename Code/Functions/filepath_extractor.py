# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import os
import re
from itertools import chain

class FilePathExtractor:
    """
    A class for reading and processing files in a directory with a specific format.
    """

    def __init__(self, directory_path: str, format_type: str) -> None:
        """
        Initialize the FilePathExtractor with a directory path and file format.
        
        Args:
        - directory_path: Path to the directory to scan for files.
        - format_type: File format (extension) to filter files, e.g., ".tif, png, jpg, ect".
                       If not found, raises an error and suggests changing the format.

        Import module:
        - from Functions.filepath_extractor import FilePathExtractor

        Example:
        - obj = FilePathExtractor(file_path, "tif")
          1. files_name = obj.filesname           # List of filename in the directory with the specified extension
          2. folders = obj.folders_path           # List of folders path where the files are located
          3. files_path = obj.all_files_path      # List of full files path for the files
          4. subfoldersname = obj.subfoldersname  # List of subfolders name within the directory
        """

        # Ensure format_type starts with a dot for consistency
        if not format_type.startswith("."):
            format_type = "." + format_type

        # Initialize class attributes
        self.files: list[str] = []  # Stores the names of all files matching the specified format.
        self.full_path: list[str] = []  # Stores the full paths of all matching files.
        self.folder_path: list[str] = []  # Stores the unique folders path containing the files.
        self.subfolder: list[list[str]] = []  # Stores the names of subfolders for each directory.
        self.format_type: str = format_type  # Stores the file format to filter (e.g., ".tif").
        self.directory_path: str = directory_path  # Stores the root directory path to scan.

        # Check if directory exists
        if not os.path.isdir(directory_path):
            raise ValueError(f"The directory '{directory_path}' does not exist.")

        self._scan_directory()  # Perform the directory scanning process.

        # Check if any files with the specified format were found
        if not self.files:
            raise ValueError(
                f"No files with the format '{format_type}' found in '{directory_path}'. "
                "Please check the format or directory path and try again."
            )

    def _scan_directory(self) -> None:
        """
        Scan the directory and its subdirectories to collect files, paths, and subfolder names.
        """
        for root, subfolder_name, files_name in os.walk(self.directory_path):  # Traverse the directory tree
            root = root.replace("\\", "/")  # Replace backslashes with forward slashes for cross-platform compatibility

            for file in files_name:
                if file.endswith(self.format_type):  # Check if the file ends with the specified format
                    self.files.append(file)  # Append the file name to the files list

                    if root not in self.folder_path:  # Check if the root folder is not already in the folder_paths list
                        self.folder_path.append(root)  # If not, append the root folder to the folder_paths list

                    self.full_path.append(os.path.join(root, file).replace("\\", "/"))  # Append the full file path

                    # Ensure subfolder names are unique and non-empty
                    if subfolder_name and subfolder_name not in self.subfolder:
                        self.subfolder.append(subfolder_name)  # Append subfolder names to subfolders list

    def _natural_sort_key(self, s: str) -> list:
        """
        Generate a key for natural sorting by splitting strings into numeric and non-numeric parts.
        
        Args:
        - s: String to generate sort key for.
        
        Returns:
        - List of parts where numbers are converted to integers for proper numerical sorting.
        """
        # Split the string into parts: non-digits and digits
        parts = re.split(r'(\d+)', s)
        # Convert numeric parts to integers, keep non-numeric parts as strings
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    @property
    def all_files_path(self) -> list[str]:
        """
        Retrieve all full files path for files with the specified format, sorted naturally.

        Returns:
        - List of full files path.
        """
        return sorted(self.full_path, key=self._natural_sort_key)

    @property
    def filesname(self) -> list[str]:
        """
        Retrieve the list of filenames, sorted naturally.

        Returns:
        - List of filenames.
        """
        return sorted(self.files, key=self._natural_sort_key)

    @property
    def folders_path(self) -> list[str]:
        """
        Retrieve the list of folders path containing the files, sorted naturally.

        Returns:
        - List of folders path.
        """
        return sorted(self.folder_path, key=self._natural_sort_key)

    @property
    def subfoldersname(self) -> list[str]:
        """
        Retrieve a flattened list of subfolders name, sorted naturally.

        Returns:
        - Flattened list of subfolders name.
        """
        return sorted(list(chain.from_iterable(self.subfolder)), key=self._natural_sort_key)