def filepath_creation(n_channel, type=1):
    """"
    Creates and reutrns the extension for openephys filenames
    """
    if type == 1:
        file_extension = "100_CH" + str(n_channel)
    elif type == 2:
        file_extension = "100_" + str(n_channel + 20)
    else:
        file_extension = "100_CH" + str(n_channel) + "_2"

    file_extension = file_extension + ".continuous"

    return file_extension

