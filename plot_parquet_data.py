import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot x vs px and y vs py for .parquet files in temp directory")
    parser.add_argument("temp_dir", help="Path to the temp directory containing .parquet files")
    parser.add_argument("--uncertainties", action="store_true", help="Plot uncertainties instead of actual coordinates")
    args = parser.parse_args()

    temp_dir = args.temp_dir
    if not os.path.exists(temp_dir):
        print(f"Directory {temp_dir} does not exist.")
        return

    files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not files:
        print(f"No .parquet files found in {temp_dir}")
        return

    for file_path in files:
        print(f"Processing {file_path}")
        try:
            df = pd.read_parquet(file_path)
            if args.uncertainties:
                required_cols = ['var_x', 'var_px', 'var_y', 'var_py', 'name']
                x_col, px_col, y_col, py_col = 'var_x', 'var_px', 'var_y', 'var_py'
                plot_type = 'variances'
            else:
                required_cols = ['x', 'px', 'y', 'py', 'name']
                x_col, px_col, y_col, py_col = 'x', 'px', 'y', 'py'
                plot_type = 'coordinates'

            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {file_path}: missing required columns")
                continue

            num_away = 14
            df = df[df['name'].str.contains(f'BPM.{num_away}R1|BPM.{num_away}L2')]

            if df.empty:
                print(f"No data for BPM.{num_away}R1 or BPM.{num_away}L2 in {file_path}")
                continue

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            axs[0].scatter(df[x_col], df[px_col], s=1, alpha=0.5)
            axs[0].set_xlabel(x_col)
            axs[0].set_ylabel(px_col)
            axs[0].set_title(f'{os.path.basename(file_path)} - {x_col} vs {px_col}')

            axs[1].scatter(df[y_col], df[py_col], s=1, alpha=0.5)
            axs[1].set_xlabel(y_col)
            axs[1].set_ylabel(py_col)
            axs[1].set_title(f'{os.path.basename(file_path)} - {y_col} vs {py_col}')

            plt.tight_layout()
            output_file = os.path.join(temp_dir, f"{os.path.basename(file_path).replace('.parquet', '')}_{plot_type}_plots.png")
            plt.savefig(output_file, dpi=150)
            plt.close()
            print(f"Saved plot to {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()