import os  # Import the os module for interacting with the operating system (e.g., file system traversal)
from itertools import chain  # Import chain for flattening nested lists

class DirectoryReader:
    """
    A class for reading and processing files in a directory with a specific format.
    """
    def __init__(self, directory_path: str, format_type: str) -> None:
        """
        Initialize the DirectoryReader with a directory path and file format.

        :param directory_path: Path to the directory to scan for files.
        :param format_type: File format (extension) to filter files, e.g., ".tif".
        """        
        self.files: list[str] = []
        self.full_path: list[str] = []
        self.folder_path: list[str] = []
        self.subfolder: list[list[str]] = []
        self.format_type: str = format_type
        self.directory_path: str = directory_path

        self._scan_directory()

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
