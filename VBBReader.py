import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from FileReader import FileReader
from VBBFile import VBBFile

class VBBReader:

    def __init__(self):
        
        # Default to big endian as VBB headers are stored in this format
        self.file_Endianness: str = '>'

        # Flag if the file is fully read or not
        self.isEoF: bool = False

        # Version of the VBB file being read. Changes how some values are stored
        self.format_version: int = 0

        # Flag whether time is in local or UTC format
        self.UTC: bool = False

        # This is the file reader
        self.reader: FileReader = None

        # This is a flag used to indicate if we've reached the point in the file where the samples are stored
        self.samples_reached: bool = False

        # This is where the data is stored once extracted from the file
        self.vbb_file: VBBFile = VBBFile()

        # This is used to more efficiently extract data from the VBB file.
        # We direct address this so the channel group in position 61 is the group with ID 61.
        self.channel_group_data: List[Dict[str, object]] = []




    def read_vbb_file(self, vbb_path):

        self.reader = FileReader(vbb_path)
        print('File opened')

        self.read_vbb_headers()
        print('Headers read')

        self.set_up_for_sample_reading()
        print('Channels parsed')

        self.read_sample_group_locations()
        print('Sample instances mapped')

        self.extract_sample_group_data()
        print('Sample group data extract')

        self.extract_channel_data()
        print('Channels extracted')

        self.correct_utc_midnight_rollovers()

        return self.vbb_file


    def read_vbb_headers(self):
        """
        This takes a VBB file and reads through it until sample data is reached. 
        It will use the headers to set flags within the VBBReader.
        """

        (headers, self.isEoF) = self.reader.read_bytes(3)
        expected_headers = np.array([0x56, 0x42, 0x42], dtype=np.uint8)

        if not np.array_equal(headers, expected_headers):
            raise Exception("Error: invalid file format")

        # The next byte contains the format of the VBB file being dealt with
        self.format_version = self.read_vbb_value('u1')
        

        # The next 4 bytes containg the flags used to notify how the file is set up
        flags = self.read_vbb_value('u4')

        # The first bit of 'flags' denotes the endianness of the file - 0 = little, 1 = big
        if flags & 0x0001 == 0x0001:
            self.file_Endianness = '>' # big endian
        else:
            self.file_Endianness = '<' # little endian

        # The second bit of 'flags' denotes the same time - 0 = local ltime, 1 = UTC
        if flags & 0x0002 == 0x0002:
            self.UTC = True
        else:
            self.UTC = False

        # The next 8 bytes contain the datetime of when the file was created
        self.vbb_file.file_created = self.read_vbb_value('datetime')
        # The next 8 bytes contain the datetime of when the file was last modified
        self.vbb_file.file_last_modified = self.read_vbb_value('datetime')


        # Read through the file until we reach data samples - this will
        # read in all of the channel/group/dictionary/EEPROM
        # information
        while not self.samples_reached:

            # Read out the next byte - this is the next section's record identifier
            section_record_ID = self.read_vbb_value('u1')

            # Parse the data from the record
            self.read_vbb_record(section_record_ID)


#### Efficient sample reading functions ####

    def set_up_for_sample_reading(self):
        # Sample reading works as follows:
        # - logic/process of each step
        # ~ the resulting data structure
        #
        #   - jump from one sample group ID to the next, noting where
        #     each instance of a sample group appears in the file.
        #   ~ a row vector of integers
        #
        #   - using the locations of each instance for each sample
        #     group, slice the byte array to extract every instance of
        #     each sample group
        #   ~ a 2D array, where each column is the raw bytes representing
        #     each instance of a sample group in the file
        #
        #   - using the locations of each channel in a sample group,
        #     slice each sample group array to get a list of all the
        #     bytes making up each instance of a channel
        #   ~ a 2D array where each column is the raw bytes
        #     representing each instance of a channel in the file
        #
        #   - transpose then typecast these arrays into the data type
        #     of the channel to get out the stored data for each
        #     channel in the file
        #   ~ a column vector of values representing the stored channel
        #     in the VBB file
        #
        # In order to achieve this we need to know the length of each
        # sample group. We calculate this using the data types of each
        # channel given in the ChannelDefinitions struct. We also need
        # to know the locations of each channel within a sample group.
        # We calculate while calculating the length of a sample group.
        #
        # To speed things up, we use direct addressing for the sample
        # group definitions (we put sample group ID 14 at position 14
        # in the GroupDefinitions array). We also use direct addressing
        # for the channels given in each GroupDefinitions entry
        # (instead of Channel ID we provide the index of that channel
        # in the ChannelDefinitions struct).


        # Go though the list of channel group definitions in the VBB
        for i in self.vbb_file.channel_group_definitions:

            groupID = i['groupID']
            channelIDs = i['channelIDs']
            numChannels = i['numChannels']

            # Channel locations within the ChannelDefinitions struct
            channelLocations = np.zeros(numChannels)

            # Length of the channel group. The 6 bytes at the start are the sample group record byte,
            # the 4 timestamp bytes and then the sample group ID byte
            groupLength = 1 + 4 + 1
            # Locations of each channel start and end within a channel group
            channelStartIndices = np.zeros(numChannels)
            channelEndIndices = np.zeros(numChannels)

            # For each of the channel IDs in the group definition, find
            # their location in the channelDefinitions struct
            #
            # While looping through each channel, we take the size of
            # each channel data type and add it to the length of the
            # current channel group. We also note where this channel
            # starts and ends within the channel group
            for j in range(numChannels):
                channelID = channelIDs[j]
                
                try:
                    channelIndex = next(
                    idx for idx, cd in enumerate(self.vbb_file.channel_definitions)
                    if cd['channelID'] == channelID
                    )
                except StopIteration:
                    raise ValueError(f"ChannelID {channelID} not found in channel definitions struct")

                # Note the location of the channel in the channel definitions struct
                channelLocations[j] = channelIndex

                # Note the start index of this channel in the channel group
                channelStartIndices[j] = groupLength + 1

                # Get the channel type from the channel definitions struct
                channelType = self.vbb_file.channel_definitions[channelIndex]['valueType']
                # Add the size of this channel to the group length
                groupLength = groupLength + self.get_vbb_value_type_sizes(channelType)

                # # Note the end index of this channel in the channel group
                channelEndIndices[j] = groupLength
                j += 1


            # If the channel_group_data array isn't long enough then we append empty structs until it is
            while len(self.channel_group_data) < groupID:
                self.channel_group_data.append({})

            # Add these calculated channel variables to the channelGroupData struct
            self.channel_group_data.insert(groupID, {
                'channelLocations': channelLocations,
                'channelStartIndex': channelStartIndices,
                'channelEndIndex': channelEndIndices,
                'groupLength': groupLength,
                'instanceLocations': [],
                'instanceData': []
            })


    def read_sample_group_locations(self):
        """
        This function reads through the file, jumping from each
        sample group record ID byte to the next, making a note of the
        group ID for each sample and where it is.
        """

        while not self.isEoF:

            record_ID = self.read_vbb_value('u1')     # uint8

            if record_ID != 9:
                # If this record isn't a sample group then we need to parse it
                self.read_vbb_record(record_ID)
                continue

            
            # Now read the ID of the sample group. Skip forward past
            # the 4 timestamp bytes to read this
            self.isEoF = self.reader.advance_through_file(4)
            groupID = self.read_vbb_value('u1')    # uint8

            # Get the struct for the group ID
            groupLength = self.channel_group_data[groupID]['groupLength']

            # Note where this sample group appears in the file. We want the location of the record ID,
            # so go back enough bytes to cover the read group ID, 4 timestamp bytes and the record ID byte
            self.channel_group_data[groupID]['instanceLocations'].append(self.reader.read_point - 6)

            # Advance through the file to the next sample group location - use the length of this sample group to do so
            # but subtract 6 as the group length includes the group ID byte we've just read, the 4 timestamp bytes and
            # the record ID byte
            self.isEoF = self.reader.advance_through_file(groupLength - 6)


    def extract_sample_group_data(self):
        """
        This function can only run once the positions of each sample group have been found in the byte array.
        It takes the positions of each instance of each sample group in the file and pulls out the bytes that make up the data.
        """

        # Convert the binary data into a numpy array of uint8 to use the advanced indexing features
        data_array = np.frombuffer(self.reader.data, dtype=np.uint8)

        # Let's manipulate the VBB byte array into sample group sections
        for channel_group in self.channel_group_data:

            # Make sure there's a channel group at this position in the struct (this accounts for our direct
            # addressing of sample groups). It also solves issues if we have a channel group defined and no
            # samples were written to the VBB file.
            if len(channel_group) == 0 or len(channel_group['instanceLocations']) == 0:
                continue

            # The group length plus the timestamp at the start
            sectionLength = channel_group['groupLength']

            # Get the row vector for the start index of each instance of this sample group in the byte array
            startLocations = channel_group['instanceLocations']

            # Create a column vector representing the position of each byte in the sample group
            offsets = np.arange(sectionLength).reshape(-1,1)

            # Generate the index matrix using implicit expansion. This creates a matrix where every column represents an
            # instance of a sample group appearing in the byte array. Each row represents the index of each byte that appears
            # in the sample group.
            indices = startLocations + offsets

            # Index into the VBB byte array using the indices matrix. This effectively takes each element in the indices
            # matrix and replaces its value with the corresponding value at that index in the byte array.
            # dataMatrix = self.reader.data(indices)
            dataMatrix = data_array[indices]

            # Add each set of data to a struct
            channel_group['instanceData'] = dataMatrix


    def extract_channel_data(self):
        """
        This function must run after extract_sample_group_data. It takes
        the sample group byte arrays as indexed from the byte array,
        slices out each channel, then parses the data. We then put
        each channel's data along with the corresponding timestamps
        into the channel_definitions struct in a VBBFile object.
        """

        # Loop through each Sample Group definition
        for group_data in self.channel_group_data:

            # Make sure there's data in here
            if len(group_data) == 0 or len(group_data['instanceLocations']) == 0:
                continue

            # Extract the timestamp data first. It runs from bytes 2-5 (remeber Python is zero indexed unlike MATLAB)
            timestampDataBytes = group_data['instanceData'][1:5]

            # Reshape into a single column vector
            timestampDataBytes = timestampDataBytes.T

            # Cast it into the correct data type - timestamps are uint32
            timestamp_type_code = self.file_Endianness + 'u4'
            timestampData = np.ascontiguousarray(timestampDataBytes).view(dtype=timestamp_type_code)
            # Flatten the array into a single 1xn array
            timestampData = timestampData.reshape(-1)

            # Convert these timestamps to seconds (cast to a double array first to avoid truncation)
            # They are stored in 100us steps so we divide by 10,000 (or multiply by 1e-4) to get seconds
            timestampData = timestampData * 1e-4

            # For each channel in this sample group extract the data, typecast it, then add it to that ChannelDefinition's data array
            channel_index = 0
            for channel_location in group_data['channelLocations']:
                # Make sure the channel location is an integer
                channel_location = int(channel_location)

                # Extract information about this channel from the structs. We need the channel data type and the start
                # and end indices of the channel in the sample group byte array
                channelDataType = self.vbb_file.channel_definitions[channel_location]['valueType']

                channelScale = self.vbb_file.channel_definitions[channel_location]['scale']
                channelOffset = self.vbb_file.channel_definitions[channel_location]['offset']

                startIndex = int(group_data['channelStartIndex'][channel_index]) - 1 # Remeber Python is zero indexed
                endIndex = int(group_data['channelEndIndex'][channel_index]) 

                # Extract the channel bytes from the sample group instances
                channelDataBytes = group_data['instanceData'][startIndex:endIndex] # groupData.instanceData(startIndex:endIndex, :);
                # % Reshape into a single column vector
                channelDataBytes = channelDataBytes.T

                # Cast it into the correct data type
                channel_type_code = self.file_Endianness + channelDataType
                channel_data = np.ascontiguousarray(channelDataBytes).view(dtype=channel_type_code)
                # Flatten the array into a single 1xn array
                channel_data = channel_data.reshape(-1)


                # % Convert the channel into a double array to prevent any rounding error issues
                # % when applying the scale and offset
                # channel_data = double(channel_data);

                # Apply the channel scale and offset to each entry
                channel_data = (channel_data * channelScale) + channelOffset

                # % Put the extracted data into the end of the channel definition array
                self.vbb_file.channel_definitions[channel_location]['timestamps'] = timestampData
                self.vbb_file.channel_definitions[channel_location]['data'] = channel_data
                channel_index += 1


    def correct_utc_midnight_rollovers(self):
        """
        This function goes through the timestamps for every channel and corrects any midnight rollovers
        that may have occurred
        """

        for channel_definition in self.vbb_file.channel_definitions:
            
            timestamps = channel_definition['timestamps']
            data = channel_definition['data']

            # If the channel is empty then don't bother performing this step
            if len(timestamps) == 0:
                continue

            timestamps = self.correct_utc_mightnight_rollover_for_channel(timestamps)

            # If we're dealing with the time channel then we need to also correct the data inside the channel
            # as well as the timestamps
            if channel_definition['shortName'] == "time":
                data = self.correct_utc_mightnight_rollover_for_channel(data)


            # Now reorder the data by timestamp to account for any samples that were written to the file
            # out-of-order. 
            #
            # Get the indices of the sorted timestamp array to apply to both the timestamps and data
            sorted_indices = np.argsort(timestamps)
            timestamps = timestamps[sorted_indices]
            data = data[sorted_indices]


    def correct_utc_mightnight_rollover_for_channel(self, timestamps):
        """
        This performs the actual timestamp corrections
        """

        # For files that go over midnight UTC, the timestamp value
        # will reset to zero. Thus, we need to find where this jump
        # happens and add a day's worth of seconds to each
        # subsequent timestamp
        timeDiffs = np.diff(timestamps)

        # The rollover point will be 0 - 86,400 (there are 86,400
        # seconds in a day). Put in 1s leeway in case a sample was
        # missed either side
        timeJumps = np.where(timeDiffs < -86399)[0]

        # For each time jump, add 86,400 to the remaining timestamps
        for jumpIndex in timeJumps:
            timestamps[jumpIndex + 1:] += 86400

        return timestamps


#### Simple VBB Creation

    def create_simple_vbb(self):
        """
        This function simplifies a VBBFile object by combining channels recorded at the same frequency
        and aligning them by timestamp.
        """

        recorded_frequencies = np.array([])
        same_frequency_channels: List[Dict[str, object]] = []

        simple_vbb = {} #: List[Dict[str, object]] = []

        # Loop through each channel and estimate its frequency
        for channel_definition in self.vbb_file.channel_definitions:

            if channel_definition['timestamps'].size == 0:
                continue    # There is no data in the channel
            elif channel_definition['timestamps'].size == 1:
                temp_frequency = 0      # if there is only 1 entry then set the frequency to 0Hz
            else:
                # This is the average time between samples in seconds
                temp_frequency = ( channel_definition['timestamps'][-1] - channel_definition['timestamps'][0] )/ channel_definition['timestamps'].size
                # Turn this into frequency
                temp_frequency = round(1 / temp_frequency)

            
            frequency_index = np.where(recorded_frequencies == temp_frequency)[0]
            
            if frequency_index.size == 0:
                # If we haven't noted this frequency yet then add it to the list
                recorded_frequencies = np.append(recorded_frequencies, temp_frequency)
                same_frequency_channels.append({
                    'frequency': temp_frequency,
                    'channelDefinitions': [channel_definition]
                })
            else:
                # We've already noted this frequency so just add the channel to this group
                same_frequency_channels[frequency_index[0]]['channelDefinitions'].append(channel_definition)
                

        # Now go through each group of same-frequency channels in turn and align their samples
        index = 0
        for frequency_group in same_frequency_channels:
            [aligned_channels, aligned_timestamps] = self.align_channels(frequency_group)
            frequency = recorded_frequencies[index]

            # Add the GNSS timestamps (aligned_timestamps) as a separate array to the simple vbb
            gnss_time_name = 'time (GNSS)'
            gnss_time_channel = ({'name': 'time (GNSS)', 'units': 's', 'data': aligned_timestamps})
            aligned_channels.append(gnss_time_channel)

            field_name = f'channels_{frequency}Hz'
            simple_vbb[field_name] = aligned_channels

            index += 1

        return simple_vbb

        
    def align_channels(self, same_frequency_channels):
        """
        This function takes a group of channels that all were recorded at the same frequency and aligns their 
        timestamps to produce a group of channels with a single timestamp array.
        Samples with no data are filled with NaN values
        """
        numChannels = len(same_frequency_channels['channelDefinitions'])

        if numChannels == 0:
            print('Error: unable to create simple VBB file, no channels loaded')
            return None, None

        # Collect all timestamps from all channels
        all_timestamps = np.array([], dtype=float)

        for channel in same_frequency_channels['channelDefinitions']:
            channel_timestamps = np.array(channel['timestamps'], dtype=float)
            all_timestamps = np.concatenate((all_timestamps, channel_timestamps))

        # Get unique sorted timestamps
        all_timestamps = np.unique(all_timestamps)

        num_timestamps = len(all_timestamps)

        aligned_channels = []

        for channel in same_frequency_channels['channelDefinitions']:
            channel_name = channel['shortName']
            channel_units = channel['units']
            channel_timestamps = np.array(channel['timestamps'], dtype=float)
            channel_data = np.array(channel['data'], dtype=float)

            # Initialize aligned_channel_data with NaNs
            aligned_channel_data = np.full(num_timestamps, np.nan)

            # Find indices of channel_timestamps in all_timestamps
            indices = np.searchsorted(all_timestamps, channel_timestamps)

            # Ensure the timestamps match
            if not np.array_equal(all_timestamps[indices], channel_timestamps):
                print(f"Warning: Timestamps in channel '{channel_name}' do not match all_timestamps.")
                continue  # Skip this channel if timestamps do not match

            # Insert data into aligned_channel_data at the indices
            aligned_channel_data[indices] = channel_data

            aligned_channel_struct = {
                'name': channel_name,
                'units': channel_units,
                'data': aligned_channel_data
            }

            aligned_channels.append(aligned_channel_struct)

        return aligned_channels, all_timestamps.tolist()
      


#### Data parsing functions ####

    def read_vbb_record(self, vbb_record_ID):
        """
        This function parses most records from a VBB file. It does not parse sample groups, these are dealt with separately to make file reading faster.
        """
        # Before a record is stored in a VBB there is a single byte identifying
        # the type of record that comes after. They are as follows:
        # 
        # **RecordType (Number stored)**
        #   GroupDefinition (5)
        #   DictionaryItem (6)
        #   ChannelDefinition (7)
        #   ChannelGroupDefiniton (8)
        #   SampleGroup (9)
        #   BinaryDump (13)
        #   
        # Inputs:
        # vbbRecordID - integer

        if vbb_record_ID == 5:

            # A group definition record is written as follows:
            # byte 1 - 05 (channel group definition identifier)
            # byte 2 - the group ID
            # n bytes - group name string (7 bit encoded length at start)

            #---------------------------------------------#

            # We've already read the first byte in order to have gotten to this
            # point, so the next entry is the group ID
            groupDef_GroupID = self.read_vbb_value('u1')  # uint8
            groupDef_Name = self.read_vbb_string()

            # # Add this new entry to the file's group definitions
            self.vbb_file.group_definitions.append({   
                'groupID': groupDef_GroupID,
                'groupName': groupDef_Name
            })

        elif vbb_record_ID == 6:
            # A dictionary record is written as follows:
            # byte 1 - 06 (Dictionary item identifier)
            # byte 2 - group ID
            # n bytes - dictionary item name string (7 bit encoded length at start)
            # n + 1 byte - value type identifier
            # m bytes - the value of the dictionary item

            #---------------------------------------------#

            # We've already read the first byte in order to have gotten to this
            # point, so the next entry is the group ID
            dictItem_GroupID = self.read_vbb_value('u1')   # uint8

            # When parsing VBB strings, the parser function looks for the
            # bytes that represent the length of that string. So we don't
            # need to manually look for the string length
            dictItem_Name = self.read_vbb_string()

            # Parse the next byte to find out what type of object this
            # dictionary item is (int, single etc)
            dictItem_ValueTypeByte = self.read_vbb_value('u1')    # uint8
            dictItem_ValueType =  self.parse_vbb_value_type(dictItem_ValueTypeByte)

            # Read the dictionary item itself
            dictItem_Value = self.read_vbb_value(dictItem_ValueType)

            # Add this new entry to the file's dictionary
            self.vbb_file.dictionary.append({   
                'name': dictItem_Name,
                'value': dictItem_Value,
                'valueType': dictItem_ValueType,
                'groupID': dictItem_GroupID
            })

        elif vbb_record_ID == 7:
            # A channel definition record is written as follows:
            # byte 1 - 07 (channel definition item identifier)
            # bytes 2 and 3 - the channel ID
            # byte 4 - group to which the channel is assigned
            # n bytes - channel short name string (7 bit encoded length at start)
            # m bytes - channel long name string (7 bit encoded length at start)
            # p bytes - channel units string (7 bit encoded length at start)
            # p + 1 - channel data type
            # p + 2 - channel scale (as double)
            # p + 10 - channel offset (as double)
            # p + 11 - channel meta data string (7 bit encoded length at start)

            #---------------------------------------------#

            channelDef_ID = self.read_vbb_value('u2')     # uint16
            channelDef_GroupID = self.read_vbb_value('u1')    # uint8
            channelDef_ShortName = self.read_vbb_string()
            channelDef_LongName = self.read_vbb_string()
            channelDef_Units = self.read_vbb_string()

            channelDef_ValueTypeByte = self.read_vbb_value('u1')    # uint8
            channelDef_ValueType = self.parse_vbb_value_type(channelDef_ValueTypeByte)

            channelDef_Scale = self.read_vbb_value('f8')  # double
            channelDef_Offset = self.read_vbb_value('f8')  # double
            channelDef_Metadata = self.read_vbb_string()


            # For some of the channel scales, we use decimal values which cannot be accurately
            # represented in binary format. For these we will manually change the values to be
            # 'correct'.
            if round(channelDef_Scale,3) == 0.001:
                channelDef_Scale = 0.001
            elif round(channelDef_Scale,1) == 3.6:
                channelDef_Scale = 3.6
            
            # Add this new entry to the file's channel definitions map
            self.vbb_file.channel_definitions.append({ 
                'channelID': channelDef_ID,
                'groupID': channelDef_GroupID,
                'shortName': channelDef_ShortName,
                'longName': channelDef_LongName,
                'units': channelDef_Units,
                'valueType': channelDef_ValueType,
                'scale': channelDef_Scale,
                'offset': channelDef_Offset,
                'metaData': channelDef_Metadata,
                'timestamps': [],
                'data': []
            })

        elif vbb_record_ID == 8:
            # A channel group definition record is written as follows:
            # byte 1 - 08 (channel group definition identifier)
            # byte 2 - channel group ID
            # bytes 2 and 3 - number of channels in the group
            # each subsequent pair of bytes is a channel until we've read as many channels as defined in bytes 2 and 3

            #---------------------------------------------#

            # We've already read the first byte in order to have gotten to this point
            # so we're reading the group ID
            channelGroup_ID =  self.read_vbb_value('u1')    # uint8
            channelGroup_NumChannels = self.read_vbb_value('u2')     # uint16

            # Create the channel array that we will fill with channel IDs
            channelGroup_ChannelIDArray = np.zeros((channelGroup_NumChannels, 1), dtype=np.uint16)


            # Now read out the list of channel IDs that are in this
            # group. Their order in this list is the order samples
            # will appear later on in the file
            for i in range(channelGroup_NumChannels):
                channelGroup_ChannelIDArray[i, 0] = self.read_vbb_value('u2')     # uint16
            

            self.vbb_file.channel_group_definitions.append({
                'groupID': channelGroup_ID,
                'numChannels': channelGroup_NumChannels,
                'channelIDs': channelGroup_ChannelIDArray
            })   

        elif vbb_record_ID == 9:
            # A sample group record is written as follows:
            # byte 1 - 09 (sample group definition identifier)
            # bytes 2,3,4 and 5 - the timestamp for this group (uint32)
            # byte 6 - the group ID for this sample group
            # The rest of the bytes are each of the channels as defined in the
            # group definition

            #---------------------------------------------#

            # Reading each sample group out byte by byte is far too
            # slow. So, we don't read individual records in with
            # this function.
            self.samples_reached = True
            # Go back one sample in the file array to the start of
            # the record
            self.isEoF = self.reader.advance_through_file(-1)
        
        elif vbb_record_ID == 13:
            # A binary dump record is written as follows:
            # byte 1 - 13 (binary dump item identifier)
            # bytes 2 and 3 - block type (0000, main eeprom 8K, 0001 module settings eeprom dump 8K, 0002 ADAS lane dumpe 132K)
            # byte 4 - length of binary dump name (7 bit encoded)
            # n bytes - binary dump name
            # n + 1 - data type
            # n + 2 - length of data block
            # m bytes - data

            #---------------------------------------------#

            # We've already read the first byte in order to have gotten to this point
            # so we're reading the block type
            binDump_BlockType = self.read_vbb_value('u2')     # uint16

            # When parsing VBB strings, the parser function looks for the
            # bytes that represent the length of that string. So we don't
            # need to manually look for the string length
            binDump_Name = self.read_vbb_string()

            # Parse the next byte to find out what type of object this
            # dimary dump is - it should be a byte array
            binDump_ValueTypeByte = self.read_vbb_value('u1')    # uint8
            binDump_ValueType = self.parse_vbb_value_type(binDump_ValueTypeByte)

            # Read the binary dump item itself
            binDump_Value = self.read_vbb_value(binDump_ValueType)


            # Add this new entry to the Binary Dump
            self.vbb_file.binary_dump.append({
                'name': binDump_Name,
                'value': binDump_Value,
                'valueType': binDump_ValueType,
                'blockType': binDump_BlockType
            })

        else:
            # We have an unexpected VBB record header type. Set the
            # file reader to the end of the file and only read the
            # data up to this point
            self.reader.close_file()
            raise Exception("Error: This file contains an unexpected VBB record type. No data will be loaded past this point: " + str(vbb_record_ID))


    def read_vbb_value(self, data_type):
        """
        Takes a standard data type and parses that out of the binary file at the current read position.
        """
        if data_type == "u1":
            return self.read_primitive(self.file_Endianness + 'u1', 1)
        elif data_type == "i1":
            return self.read_primitive(self.file_Endianness + 'i1', 1)
        elif data_type == "u2":
            return self.read_primitive(self.file_Endianness + 'u2', 2)
        elif data_type == "i2":
            return self.read_primitive(self.file_Endianness + 'i2', 2)
        elif data_type == "u4":
            return self.read_primitive(self.file_Endianness + 'u4', 4)
        elif data_type == "i4":
            return self.read_primitive(self.file_Endianness + 'i4', 4)
        elif data_type == "f4":
            return self.read_primitive(self.file_Endianness + 'f4', 4)
        elif data_type == "u8":
            return self.read_primitive(self.file_Endianness + 'u8', 8)
        elif data_type == "i8":
            return self.read_primitive(self.file_Endianness + 'i8', 8)
        elif data_type == "f8":
            return self.read_primitive(self.file_Endianness + 'f8', 8)
        elif data_type == "datetime":
            return self.read_vbb_datetime()
        elif data_type == "string":
            return self.read_vbb_string()
        elif data_type == "byteArray":
            return self.read_vbb_byte_array()


    def read_primitive(self, data_type, n_bytes):
        """
        This function reads primitive types from a byte array
        """

        # Read the bytes out of the file
        (byte_Array, self.isEoF) = self.reader.read_bytes(n_bytes)
        # Convert the bytes into the correct format
        #parsed_Value = np.ascontiguousarray(byte_Array).view(dtype=data_type)
        parsed_Value = byte_Array.view(data_type)[0]

        return parsed_Value


    def read_vbb_string(self):
        """
        This function reads in a variable length string from a VBB file.
        """
        # Strings are stored in VBB files as <length><string data>
        #   length - a 7bit encoded integer and is the total number of bytes
        #            used to encode the string (NOT the string length)
        #   string data - a UTF-8 encoded string 
        string_as_bytes = self.read_vbb_byte_array()

        string = string_as_bytes.tobytes().decode('utf-8')
        return string


    def read_vbb_byte_array(self):
        """
        This function reads a byte array from a VBB file and puts it in the correct order
        based on the endianness of the file.
        """
        # Byte arrays are stored as <length><bytes>
        #   length - a 7bit encoded integer and is the total number of bytes
        #            used to encode the string (NOT the string length)
        #   bytes - the byte data

        # Get the length of the array
        array_length = self.read_7bit_encoded_int()

        # Read the array out of the file
        (byte, self.isEoF) = self.reader.read_bytes(array_length)

        return byte


    def read_7bit_encoded_int(self):
        # Use the endianness of the file to correctly parse the 7-bit encoded 
        # int length from the first byte

        # Big endian logic
        if self.file_Endianness == '>':
            integer = 0

            while True:
                byte = self.read_primitive(self.file_Endianness + 'u1', 1)  # uint8
                integer = integer << 7
                # Mask the last 7 bits and add them to the integer
                integer = integer + (byte & int(0x7F))

                # If the most significant bit is not set then the full integer has been read
                if (byte & 0x80) == 0:
                    break
            
            return integer
        # Little endian logic
        elif self.file_Endianness == '<':
            integer = 0
            shift = 0

            while True:
                byte = self.read_primitive(self.file_Endianness + 'u1', 1)  # uint8

                # Mask the 7 bits and shift them according to current position
                encoded_int_value = encoded_int_value + ((byte & 0x7F) << shift)

                # Check if MSB is set; if not, stop
                if (byte & 0x80) == 0:
                    break

                # Increment the shift value by 7
                shift = shift + 7
            
            return integer
        else:
            raise Exception("Error: unsupported endianness" + str(self.file_Endianness))


    def read_vbb_datetime(self):
        """
        This function takes a VBB file that has had the headers read into
        memory and, based on the Format Version that's been read, will extract
        the datetime from the next 8 bytes. 
        """

        if self.format_version == 1:

            # In format v1 the time data is stroed as year, month, day, hours, minutes, seconds

            year = self.read_primitive(self.file_Endianness + 'u2', 2)  # uint16
            month = self.read_primitive(self.file_Endianness + 'u1', 1)   # uint8
            day = self.read_primitive(self.file_Endianness + 'u1', 1)       
            hour = self.read_primitive(self.file_Endianness + 'u1', 1)      
            minute = self.read_primitive(self.file_Endianness + 'u1', 1)
            second = self.read_primitive(self.file_Endianness + 'u1', 1)

            return datetime(year = year, month = month, day = day, hour = hour, minute = minute, second = second)

        elif self.format_version == 2:

            # In format v2 the time data is stored in ticks - 100ns increments since 01-01-0001
            # The upper 2 bits contain metadata on the timezone being used for the datetime

            # Mask out the upper 2 metadata bits to get the raw time value
            raw_time_value = int(self.read_primitive(self.file_Endianness + 'u8', 8))   # uint64
            ticks = raw_time_value & 0x3FFFFFFFFFFFFFFF

            # 1 tick = 100 nanoseconds, therefore 1 second = 10^7 ticks
            # 1 day = 24 * 60 * 60 seconds = 86400 seconds
            seconds = ticks / 1e7
            days = seconds / 86400

            # The base for Python's datetime is 01-01-0001 which is the same as a VBB file
            parsed_datetime = datetime(1,1,1) + timedelta(days = days)

            # Calculate the time of day
            seconds_today = seconds % 86400
            hours = int(seconds_today // 3600)
            minutes = int((seconds_today % 3600) // 60)
            seconds_remaining = seconds_today % 60

            # Set the time of day
            parsed_datetime = parsed_datetime.replace(hour = hours, 
                                                      minute = minutes, 
                                                      second = int(seconds_remaining))
                                                      
            # Determine the DateTimeKind based on the upper 2 bits
            if raw_time_value & 0xC000000000000000 == 0x8000000000000000:
                # We're in local time
                parsed_datetime = parsed_datetime.astimezone()  # Local timezone
            elif raw_time_value & 0xC000000000000000 == 0:
                # We're in UTC time
                parsed_datetime = parsed_datetime.replace(tzinfo=datetime.timezibe.UTC)
            else:
                # Unspecified or floating
                parsed_datetime = parsed_datetime.replace(tzinfo=None)

            return parsed_datetime
        else:
            raise Exception("Error: unsupported VBB file format" + str(self.format_version))



#### Data type conversion from VBB to Numpy

    def parse_vbb_value_type(self, vbb_value):
        # This function holds definitions for the type of value stored in a VBB
        # file.
        #
        # Before a value is stored in a VBB there is a single byte identifying
        # the type of value that comes after. They are as follows:
        #    
        #  **ValueType (Number stored)**
        #    None (0)
        #    Byte (1)
        #    UInt16 (2)
        #    Int16 (3)
        #    UInt32 (4)
        #    Int32 (5)
        #    UInt64 (6)
        #    Int64 (7)
        #    Single (8)
        #    Double (9)
        #    Time (10)
        #    DateTime (11)
        #    String (12)
        #    ByteArray (13) - not present in the documentation might need this later though

        # These are converted into the nomenclature that NumPy uses
        if vbb_value == 0:
            return 'None'
        elif vbb_value == 1:
            return 'u1'
        elif vbb_value == 2:
            return 'u2'
        elif vbb_value == 3:
            return 'i2'
        elif vbb_value == 4:
            return 'u4'
        elif vbb_value == 5:
            return 'i4'
        elif vbb_value == 6:
            return 'u8'
        elif vbb_value == 7:
            return 'i8'
        elif vbb_value == 8:
            return 'f4'
        elif vbb_value == 9:
            return 'f8'
        elif vbb_value == 10:
            return 'time'
        elif vbb_value == 11:
            return 'datetime'
        elif vbb_value == 12:
            return 'string'
        elif vbb_value == 13:
            return 'byteArray'
        else:
            raise Exception('Unknown VBBValueType -' + vbb_value)


    def get_vbb_value_type_sizes(self, value_type):
        """
        This function takes a string representing a VBB value type and returns the number of bytes
        that said type comprises
        """

        if value_type == "u1" or value_type == "i1":
            return 1
        elif value_type == "u2" or value_type == "i2":
            return 2
        elif value_type == "u4" or value_type == "i4" or value_type == "f4":
            return 4
        elif value_type == "u8" or value_type == "i8" or value_type == "f8":
            return 8
        else: 
            raise Exception("Error: Unknown VBB value type " + str(value_type))
