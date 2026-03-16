# Crystal23_Utility
Python codes for postporocessing , extracting and compling information from Crystal DFT code output


1. crystal23_to_extxyz.py
   Crystal23 raw output → Extended XYZ converter for ML training.

    Reads: FORCES.DAT, SCFOUT.LOG, INPUT, OUTPUT, opt[a/c]* files
    Writes: Extended XYZ file specified via command-line argument

    Usage:
    python crystal23_to_extxyz.py <output_file.xyz>
