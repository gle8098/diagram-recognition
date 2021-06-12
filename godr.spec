# -*- mode: python ; coding: utf-8 -*-
import platform

block_cipher = None
data_files = [
    ('src/godr/frontend/ui', 'godr/frontend/ui'),
    ('src/godr/backend/models', 'godr/backend/models')
]
exclude_modules = ['PyQt5.QtQuickWidgets', 'PyQt5.QtQuick', 'PyQt5.QtNfc',
                   'PyQt5.QtNetwork', 'PyQt5.QtWebChannel', 'PyQt5.QtXmlPatterns',
                   'PyQt5.QtDesigner', 'PyQt5.QtLocation', 'PyQt5.QtQuick3D',
                   'PyQt5.QtTest', 'PyQt5.QtQml']

exe_name = 'godr-app'
icon_name = None

if platform.system() == 'Windows':
    exe_name = 'godr.exe'
    icon_name = 'data\\icons\\icon.ico'
elif platform.system() == 'Darwin':
    icon_name = 'data/icons/icon.icns'
    # exe_name remains default

a = Analysis(['src/godr/frontend/app_ui.py'],
             pathex=[],
             binaries=[],
             datas=data_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=exclude_modules,
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name=exe_name,
          icon=icon_name,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='godr')
app = BUNDLE(coll,
             name='godr.app',
             icon=icon_name,
             bundle_identifier=None,
             info_plist={'CFBundleName': 'Распознавание го диаграм'})
