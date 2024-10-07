import numpy as np


class FileReader:
    """
    A general class used to read and move through a binary file
    """


    def __init__(self, filePath):
        """
        Opens a binary file and stores it in memory
        """

        # Use NumPy to read the file and store it as an array
        self.vbb_file_path: str = filePath
        self.data = np.fromfile(filePath, dtype=np.uint8)
        self.read_point: int = 0



    def read_bytes(self, n_bytes):
        """
        Reads a specific number of bytes from the file and advances the read point. Will return a flag if the end of the file is reached.
        """

        # Check if the requested bytes go beyond the end of the data
        if self.read_point + n_bytes >= len(self.data):
            # Read the remaining bytes until the end of the file
            result = self.data[self.read_point:]
            self.read_point = len(self.data)  # Move the read point to the end
            eof = True  # End of file reached
        else:
            # Read the requested number of bytes
            result = self.data[self.read_point:self.read_point + n_bytes]
            self.read_point += n_bytes
            eof = False  # Not at the end yet

        return result, eof
       


    def advance_through_file(self, steps):
        """
        Advance through a file by the requested number of steps
        """

        if self.read_point + steps >= len(self.data):
            return True

        self.read_point += steps
        return False

    
    def close_file(self):
        """
        Close the file and stop reading
        """

        self.data = None
        self.read_point = 0

