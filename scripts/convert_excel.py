#!/usr/bin/env python3
"""Convert Excel file with Reddit data to CSV files.

Usage:
    python scripts/convert_excel.py /path/to/file.xlsx
    python scripts/convert_excel.py /path/to/file.xlsx --output-dir ./data
    python scripts/convert_excel.py /path/to/file.xlsx --sheets Coffee,learnprogramming
"""

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


def extract_title_from_url(url: str) -> str:
    """Extract post title from Reddit URL.

    Reddit URLs have format:
    https://www.reddit.com/r/subreddit/comments/id/title_slug

    Args:
        url: Reddit post URL

    Returns:
        Extracted and cleaned title
    """
    if not url or not isinstance(url, str):
        return ""

    parts = url.rstrip('/').split('/')

    # Find the title slug (usually last part after comments/id/)
    if len(parts) >= 7:
        title_slug = parts[-1]
        # URL decode and replace underscores/hyphens with spaces
        title = unquote(title_slug)
        title = title.replace('_', ' ').replace('-', ' ')
        # Capitalize first letter of each sentence
        title = title.strip()
        if title:
            title = title[0].upper() + title[1:]
        return title

    return ""


def convert_sheet_to_csv(
    df: pd.DataFrame,
    sheet_name: str,
    output_dir: Path
) -> Path:
    """Convert a DataFrame to CSV with correct format.

    Args:
        df: DataFrame with input_url and selftext columns
        sheet_name: Name of the sheet (used for filename)
        output_dir: Directory to save CSV file

    Returns:
        Path to created CSV file
    """
    # Create output dataframe with correct column names
    output_df = pd.DataFrame()

    # Map columns
    url_col = 'input_url' if 'input_url' in df.columns else 'url'
    text_col = 'selftext' if 'selftext' in df.columns else 'post_text'

    output_df['url'] = df[url_col]

    # Extract title from URL if no title column exists
    if 'title' in df.columns:
        output_df['title'] = df['title']
    else:
        output_df['title'] = df[url_col].apply(extract_title_from_url)

    output_df['post_text'] = df[text_col].fillna('')

    # Remove duplicates by URL
    output_df = output_df.drop_duplicates(subset=['url'], keep='first')

    # Remove rows with empty URLs
    output_df = output_df[output_df['url'].notna() & (output_df['url'] != '')]

    # Save to CSV
    output_path = output_dir / f"{sheet_name}.csv"
    output_df.to_csv(output_path, index=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Excel file with Reddit data to CSV files"
    )
    parser.add_argument("excel_file", help="Path to Excel file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for CSV files (default: current directory)"
    )
    parser.add_argument(
        "--sheets",
        type=str,
        help="Comma-separated list of sheets to convert (default: all except 'All')"
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Excel file: {excel_path}")
    xlsx = pd.ExcelFile(excel_path)

    print(f"\nFound sheets: {xlsx.sheet_names}")

    # Determine which sheets to process
    if args.sheets:
        sheets_to_process = [s.strip() for s in args.sheets.split(',')]
    else:
        # Skip 'All' sheet by default as it contains duplicates
        sheets_to_process = [s for s in xlsx.sheet_names if s.lower() != 'all']

    print(f"Processing sheets: {sheets_to_process}")

    results = []
    for sheet_name in sheets_to_process:
        if sheet_name not in xlsx.sheet_names:
            print(f"\n  Warning: Sheet '{sheet_name}' not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {sheet_name}")
        print('='*50)

        df = pd.read_excel(xlsx, sheet_name=sheet_name)
        print(f"  Rows in sheet: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        output_path = convert_sheet_to_csv(df, sheet_name, output_dir)

        # Read back to get final stats
        final_df = pd.read_csv(output_path)
        print(f"  Saved to: {output_path}")
        print(f"  Final rows (after dedup): {len(final_df)}")

        results.append({
            'sheet': sheet_name,
            'original_rows': len(df),
            'final_rows': len(final_df),
            'output_file': output_path
        })

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    for r in results:
        print(f"  {r['sheet']}: {r['original_rows']} -> {r['final_rows']} rows")

    print(f"\nTotal files created: {len(results)}")
    print("Done!")


if __name__ == "__main__":
    main()
