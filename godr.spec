# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['src/godr/frontend/app_ui.py'],
             pathex=[],
             binaries=[],
             datas=[('src/godr/frontend/ui', 'godr/frontend/ui'), ('src/godr/backend/models', 'godr/backend/models')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          name='godr.app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False )
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
             icon=None,
             bundle_identifier=None)
