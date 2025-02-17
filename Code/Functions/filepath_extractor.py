# ================================ Presented by: Reza Saadatyar (2023-2024) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import os  # Import the os module for interacting with the operating system (e.g., file system traversal)
from itertools import chain  # Import chain for flattening nested lists

class FilePathExtractor:
    """
    A class for reading and processing files in a directory with a specific format.
    """

    def __init__(self, directory_path: str, format_type: str) -> None:
        """
        Initialize the FilePathExtractor with a directory path and file format.
        
        Parameters:
        - directory_path: Path to the directory to scan for files.
        - format_type: File format (extension) to filter files, e.g., ".tif".
        
        Import module:
        - from Functions.filepath_extractor import FilePathExtractor
        
        Example:
        - obj = FilePathExtractor(file_path, "csv")
          1. file_names = obj.filenames          # List of filenames in the directory with the specified extension
          2. folder = obj.folder_paths           # List of folder paths where the files are located
          3. file_paths = obj.all_file_paths     # List of full file paths for the files
          4. subfoldernames = obj.subfoldernames # List of subfolder names within the directory
        """
        
        # Initialize class attributes      
        self.files: list[str] = [] # Stores the names of all files matching the specified format.
        self.full_path: list[str] = [] # Stores the full paths of all matching files.
        self.folder_path: list[str] = [] # Stores the unique folder paths containing the files.
        self.subfolder: list[list[str]] = [] # Stores the names of subfolders for each directory.
        self.format_type: str = format_type # Stores the file format to filter (e.g., ".tif").
        self.directory_path: str = directory_path # Stores the root directory path to scan.
        self._scan_directory()  # Perform the directory scanning process.

    def _scan_directory(self) -> None:
        for root, subfolder_name, files_name in os.walk(self.directory_path):  # Traverse the directory tree
            root = root.replace("\\", "/")  # Replace backslashes with forward slashes for cross-platform compatibility

            for file in files_name:
                if file.endswith(self.format_type):  # Check if the file ends with the specified format
                    self.files.append(file)  # Append the file name to the files list

                    if root not in self.folder_path:  # Check if the root folder is not already in the folder_paths list
                        self.folder_path.append(root)  # If not, append the root folder to the folder_paths list

                    self.full_path.append(os.path.join(root, file).replace("\\", "/"))  # Append the full file path

                    # Ensure subfolder names are unique and non-empty
                    if subfolder_name not in self.subfolder and subfolder_name != []:
                        self.subfolder.append(subfolder_name)  # Append subfolder names to subfolders list

    @property
    def all_file_paths(self) -> list[str]:
        """
        Retrieve all full file paths for files with the specified format.

        :return: List of full file paths.
        """
        return self.full_path

    @property
    def filenames(self) -> list[str]:
        """
        Retrieve the list of filenames.

        :return: List of filenames.
        """
        return self.files

    @property
    def folder_paths(self) -> list[str]:
        """
        Retrieve the list of folder paths containing the files.

        :return: List of folder paths.
        """
        return self.folder_path

    @property
    def subfoldernames(self) -> list[str]:
        """
        Retrieve a flattened list of subfolder names.

        :return: Flattened list of subfolder names.
        """
        return list(chain.from_iterable(self.subfolder))
