import os, platform, subprocess


def open_file_in_external_app(filepath):
    """
    Opens file in the user's preferred application.
    Supports Windows, MacOS, Ubuntu (maybe other distributives).
    :raise child of Exception if something goes wrong
    """

    if platform.system() == 'Darwin':  # MacOS
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':  # Windows
        os.startfile(filepath)
    else:  # Linux distributives
        subprocess.call(('xdg-open', filepath))
