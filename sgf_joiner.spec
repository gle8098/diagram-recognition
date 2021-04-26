# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['src/godr/frontend/sgf_joiner_ui.py'],
             pathex=[],
             binaries=[],
             datas=[('src/godr/frontend/ui/merge_sgf.ui', 'godr/frontend/ui/'),
                    ('src/godr/frontend/ui/__init__.py',  'godr/frontend/ui/')],
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
          name='sgf_joiner.app',
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
               name='sgf_joiner')
app = BUNDLE(coll,
             name='sgf_joiner.app',
             icon=None,
             bundle_identifier=None)
