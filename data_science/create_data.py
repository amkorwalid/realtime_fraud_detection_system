import polars as pl
from pathlib import Path
import polars.selectors as cs

file_path = Path.cwd().parent / "data" / "raw" / "dataset.csv"
print(f"Reading data from: {file_path}")

df = pl.read_csv(file_path)

df_train = df.filter(pl.col("step") < 300)
df_test = df.filter((pl.col("step") >= 300) & (pl.col("step") < 350))
df_stream = df.filter(pl.col("step") >= 350)

print(f"""
        Train shape: {df_train.shape}
        Test shape: {df_test.shape}
        Stream shape: {df_stream.shape}
    """)

# Save the datasets to csv files
df_train.write_csv(Path.cwd().parent / "data" / "offline" / "train.csv")
print(f"Train data saved to: {Path.cwd().parent / 'data' / 'offline' / 'train.csv'}")
df_test.write_csv(Path.cwd().parent / "data" / "offline" / "test.csv")
print(f"Test data saved to: {Path.cwd().parent / 'data' / 'offline' / 'test.csv'}")
df_stream.write_csv(Path.cwd().parent / "data" / "stream" / "stream.csv")
print(f"Stream data saved to: {Path.cwd().parent / 'data' / 'stream' / 'stream.csv'}")