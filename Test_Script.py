from VBBReader import VBBReader
from VBBFile import VBBFile
from datetime import datetime
import matplotlib.pyplot as plt


def plot_channel(vbb_file, channel_name):
    timestamps, data = vbb_file.get_channel_data(channel_name)
    
    if timestamps is None or data is None:
        print(f"Channel '{channel_name}' not found!")
        return

    # Plot timestamps vs data
    plt.figure(figsize=(8, 6))
    plt.plot(timestamps, data, marker="o", linestyle="-")
    plt.title(f"{channel_name}")
    plt.xlabel("GNSS Time (s)")
    plt.ylabel(channel_name)
    plt.grid(True)
    plt.show()


def list_channels(vbb_file):
    ## This is an example function which gets a list of channels in a vbb_file and allows the user to plot them ###
    
    # Display channel names
    print("Available channels:")
    channel_names = vbb_file.get_channel_names()
    for idx, name in enumerate(channel_names, 1):
        print(f"{idx}. {name}")

    # Ask the user to select a channel
    try:
        choice = int(input("Select a channel number to plot: "))
        if 1 <= choice <= len(channel_names):
            selected_channel = channel_names[choice - 1]
            plot_channel(vbb_file, selected_channel)
        else:
            print("Invalid choice.")
    except ValueError:
        print("Please enter a valid number.")


def main():

    reader = VBBReader()
    reader.read_vbb_file('ExampleBrakeStop.vbb')

    # This will read a VBB file into Python and create a dictionary object.
    # 
    # Data is stored in the 'channelDefinitions' entry. Each channel has a 'timestamps' and 'data' array.
    # 'timestamps' is the set of GNSS timestamps that each channel is written to the file with. There will also 
    # be a channel called 'time' which is UTC time and is recorded as its own channel (and will have 'timestamps'
    # with it).
    #
    # 'file_created' and 'file_last_modified' are timestamps in the LOCAL timezone and refer to the file itself
    # Inside 'dictionary' there are 'date' and 'append' timestamps. These are in UTC and refer to when the data in the 
    # file started being written and (if a recording was paused) resumed writing respectively.
    vbb_file = reader.vbb_file

    # This will 'simplify' the vbb_file object. It will calculate the frequencies of each channel, group like-frequencies together
    # and align their timestamps. 
    #
    # The output will be a dictionary with entries for each frequency group called 'channels_<x>Hz'.
    # Each entry in the frequency group contains the 'name', 'units' and 'data' for each channel.abs
    # There will be a 'time (GNSS)' channel as well. This is the combined set of timestamps from each 
    # channel in the vbb file
    simple_vbb_file = reader.create_simple_vbb()


    list_channels(vbb_file)


if __name__ == "__main__":
    main()