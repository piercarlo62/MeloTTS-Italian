import pandas as pd
import argparse

def convert_xlsx_to_txt(input_path, output_path, sheet_name=None):
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    df.to_csv(output_path, sep='|', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XLSX to pipe-delimited TXT.")
    parser.add_argument("input", help="Path to the input XLSX file")
    parser.add_argument("output", help="Path to the output TXT file")
    parser.add_argument("--sheet", help="Name of the sheet to read (optional)", default=None)

    args = parser.parse_args()
    convert_xlsx_to_txt(args.input, args.output, args.sheet)

