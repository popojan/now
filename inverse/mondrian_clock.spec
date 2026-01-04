# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Mondrian Clock Inverse Analyzer
#
# Build with: pyinstaller mondrian_clock.spec
# Or on Windows: pyinstaller mondrian_clock.spec --clean

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect OpenCV data files
opencv_datas = collect_data_files('cv2')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=opencv_datas,
    hiddenimports=[
        'cv2',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mondrian-clock-analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress if UPX is available
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console app for CLI usage
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired: 'icon.ico'
)
