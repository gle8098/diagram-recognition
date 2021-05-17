from PyQt5 import QtWidgets, uic
import pkg_resources


class WarningDialog(QtWidgets.QDialog):
    def __init__(self, path_sgf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load Qt Designer form
        form_class, _ = uic.loadUiType(pkg_resources.resource_stream('godr.frontend.ui', "warning.ui"))
        form_class().setupUi(self)

        self.findChild(QtWidgets.QLabel, 'file_name_label').setText(path_sgf)
        self.rewrite = False

    def select_yes(self):
        self.rewrite = True
        self.close()

    def select_no(self):
        self.rewrite = False
        self.close()

    def get_rewrite_status(self):
        return self.rewrite
