import os
import platform
import subprocess


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


def translate_plural(n, single, middle, multiple):
    """
    Chooses which form of plural to use. Based on https://doc.qt.io/qt-5/i18n-plural-rules.html.
    :param n: the number itself
    :param single: string to use by rule 1
    :param middle: string to use by rule 2
    :param multiple: string to use by rule 3
    """
    # todo: use Qt Linguist & its tr() function
    if n % 10 == 1 and n % 100 != 11:
        return single
    elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 > 20):
        return middle
    else:
        return multiple


class Progress:
    """
    This class is used to calculate progress on nested tasks. Here is an example:

    * <layer 0> recognize each of selected files
        * <layer 1> current file is pdf, recognize each page
            * <layer 2> several boards found on page, recognize each

    As you can see, recognition is a nested task. To properly calculate global progress, each simple task makes
    a layer for itself. Layer contains a list of two elements, ``[current, total]``, which means that the job has
    been done for ``current / total * 100`` percent.
    """
    def __init__(self, init_range=None):
        self._progress = []
        if init_range:
            self.add_layer(init_range)

    def __warn(self, msg):
        print('WARN RecognitionWorker.Progress: {}'.format(msg))

    def add_layer(self, prange):
        assert len(prange) == 2 and prange[0] <= prange[1]
        self._progress.append(list(prange))
        return len(self._progress) - 1

    def append_progress(self, layer, delta):
        assert layer < len(self._progress)
        self.set_progress(layer, self._progress[layer][0] + delta)

    def set_progress(self, layer, value):
        assert layer < len(self._progress)
        if not (0 <= value <= self._progress[layer][1]):
            self.__warn('setting progress > 1')
        self._progress[layer][0] = value

    def pop_layer(self, layer):
        assert layer < len(self._progress)
        if layer + 1 < len(self._progress):
            self.__warn('pop layer with its child')
        self._progress = self._progress[:layer]

    def calc(self):
        fraction = 0.0
        for subprogress in self._progress[::-1]:
            fraction = (subprogress[0] + fraction) / subprogress[1]
            if fraction > 1:
                self.__warn('fraction > 1')
                fraction = 1
        return int(100 * fraction)
