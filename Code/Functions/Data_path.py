import os  # Import the os module for interacting with the operating system (e.g., file system traversal)
from itertools import chain  # Import chain for flattening nested lists

class DirectoryReader:
    def __init__(self, directory_path: str, format_type: str) -> None:
        # Initialize the class attributes
        self.files: list[str] = []  # List to store filenames
        self.full_path: list[str] = []  # List to store full file paths
        self.folder_path: list[str] = []  # List to store folder paths where the files are located
        self.subfolders: list[list[str]] = []  # List to store subfolders in each directory
        self.format_type: str = format_type  # The file format type (e.g., ".tif")
        self.directory_path: str = directory_path  # The directory path to scan for files

    @property
    def all_file_paths(self) -> list[str]:
        """Property that retrieves all file paths with the specified format."""
        for root, subfolder_name, files_name in os.walk(self.directory_path):  # Traverse the directory tree
            root = root.replace("\\", "/")  # Replace backslashes with forward slashes for cross-platform compatibility

            for file in files_name:
                if file.endswith(self.format_type):  # Check if the file ends with the specified format
                    self.files.append(file)  # Append the file name to the files list

                    if root not in self.folder_path:  # Check if the root folder is not already in the folder_paths list
                        self.folder_path.append(root)  # If not, append the root folder to the folder_paths list

                    self.full_path.append(os.path.join(root, file).replace("\\", "/"))  # Append the full file path

                    # Ensure subfolder names are unique and non-empty
                    if subfolder_name not in self.subfolders and subfolder_name != []:
                        self.subfolders.append(subfolder_name)  # Append subfolder names to subfolders list

        return self.full_path  # Return the list of full file paths

    @property
    def filenames(self) -> list[str]:
        """Property that returns the list of filenames."""
        return self.files  # Return the list of filenames

    @property
    def folder_paths(self) -> list[str]:
        """Property that returns the list of directories containing the files."""
        return self.folder_path  # Return the list of folder paths

    @property
    def subfoldernames(self) -> list[str]:
        """Property that returns a flattened list of subfolder names."""
        return list(chain.from_iterable(self.subfolders))  # Flatten the list of subfolders using chain and return it