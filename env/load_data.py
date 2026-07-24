import pandas as pd

try:
    from IPython.display import display
except ModuleNotFoundError:
    def display(value):
        print(value)


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp950", "big5", "gb18030", "latin1")


def read_csv_flexible(path, **kwargs):
    """Read CSV files with common UTF and Chinese Windows encodings."""
    last_error = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs), encoding
        except UnicodeDecodeError as exc:
            last_error = exc
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs), "utf-8-replace"
    except TypeError:
        if last_error:
            raise last_error
        raise


def read_table_flexible(path, **kwargs):
    """Read CSV or Excel files while preserving the CSV encoding fallback behavior."""
    lower_path = str(path).lower()
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, **kwargs), "excel"
    return read_csv_flexible(path, **kwargs)


