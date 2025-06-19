import pandas as pd
import argparse

def convert_xlsx_to_txt(input_path, output_path, sheet_name=None, win=False, skip_first_row=False):
    print(f"Reading Excel file: {input_path}")
    print(f"Sheet: {sheet_name if sheet_name else 'First sheet (index 0)'}")
    print(f"Skip first row: {skip_first_row}")
    
    # Read the Excel file with explicit parameters
    if sheet_name is not None:
        df = pd.read_excel(input_path, sheet_name=sheet_name, header=None, dtype=str)
    else:
        df = pd.read_excel(input_path, sheet_name=0, header=None, dtype=str)
    
    print(f"DataFrame shape after reading: {df.shape}")
    
    # Ensure df is a DataFrame, not a dict
    if isinstance(df, dict):
        df = list(df.values())[0]
        print("Converted from dict to DataFrame")
    
    # Skip first row if requested
    if skip_first_row:
        print("Skipping first row...")
        df = df.iloc[1:]
        print(f"DataFrame shape after skipping: {df.shape}")
        
    # Handle Windows path conversion
    if win and len(df.columns) > 0:
        print("Converting forward slashes to backslashes in first column...")
        # First column is index 0
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.replace('/', '\\', regex=False)
        print("After path conversion:")
        print(df.head(3))
    
    # Write to file
    print(f"Writing to: {output_path}")
    df.to_csv(output_path, sep='|', index=False, header=False)
    print("Conversion completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XLSX to pipe-delimited TXT.")
    parser.add_argument("input", help="Path to the input XLSX file")
    parser.add_argument("output", help="Path to the output TXT file")
    parser.add_argument("--sheet", help="Name or index of the sheet to read (optional)", default=None)
    parser.add_argument("--win", action="store_true", help="Convert forward slashes to backslashes in first column for Windows paths")
    parser.add_argument("--skip-first-row", action="store_true", help="Skip the first row of the Excel sheet")
    
    args = parser.parse_args()
    convert_xlsx_to_txt(args.input, args.output, args.sheet, args.win, args.skip_first_row)


