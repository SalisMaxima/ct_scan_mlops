"""Convenience wrapper for downloading CT scan dataset.

This script is a simple wrapper around the download_data() function in data.py.
You can use this directly or call the function from the module.

Usage:
    python get_data.py                           # Downloads to data/raw
    python -m ct_scan_mlops.data download        # Same as above
    invoke download-data                          # Using invoke task
"""

from ct_scan_mlops.data import download_data

if __name__ == "__main__":
    download_data()
