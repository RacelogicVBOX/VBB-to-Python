from datetime import datetime
from typing import List, Dict

class VBBFile:
    """
    This class holds information that is read from a VBB file.
    It includes header information such as the Endianness and the format of the file.
    """
    
    def __init__(self):
        
        # All these fields are empty before the file is read

        # Datestamps for file creation and modification
        self.file_created: datetime = None
        self.file_last_modified: datetime = None

        # Channel group definitions
        self.group_definitions: List[Dict[str, object]] = []

        # System and setting information
        self.dictionary: List[Dict[str, object]] = []

        # Holds information about the VBB channels stored in the file.
        # Each entry is timestamped. The scale and offset have been applied
        # to the values in the data array.
        self.channel_definitions: List[Dict[str, object]] = []

        # Holds information on the specific channel IDs contained within channel groups
        self.channel_group_definitions: List[Dict[str, object]] = []

        # Holds the raw EEPROM data
        self.binary_dump: List[Dict[str, object]] = []


    # A couple of useful functions 
    def get_channel_names(self):
        return [channel["shortName"] for channel in self.channel_definitions]


    def get_channel_data(self, channel_name):
        for channel in self.channel_definitions:
            if channel["shortName"] == channel_name:
                return channel["timestamps"], channel["data"]
        return None, None