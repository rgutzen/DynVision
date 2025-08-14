import pandas as pd
import argparse
import os
import hashlib


def get_file_size_gb(file_path):
    """Get file size in GB"""
    return os.path.getsize(file_path) / (1024**3)


def hash_row(row):
    """Create a hash of a row for duplicate detection"""
    return hashlib.md5(str(row.values).encode()).hexdigest()


def process_simple(input_path, output_path):
    """Simple processing for files < 4GB"""
    print("Using simple processing (loading entire file)...")

    # Load entire CSV
    df = pd.read_csv(input_path, low_memory=False)
    original_shape = df.shape

    print(f"Original shape: {original_shape}")

    # Remove columns and duplicates
    df = df.drop(columns=["response", "class_index"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)

    # Save result
    df.to_csv(output_path, index=False)

    print(f"Reduced shape: {df.shape}")
    print(
        f"Reduction: {original_shape[0] - len(df):,} rows, {original_shape[1] - len(df.columns)} columns removed"
    )

    return len(df), original_shape[0]


def process_chunked(input_path, output_path, chunksize):
    """Chunked processing for files >= 4GB"""
    print(f"Using chunked processing (chunks of {chunksize:,} rows)...")

    # Track seen rows to avoid duplicates
    seen_hashes = set()
    first_chunk = True
    total_rows_processed = 0
    total_rows_kept = 0

    # Process file in chunks
    for chunk_num, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
    ):
        print(f"Processing chunk {chunk_num + 1}... ({len(chunk):,} rows)")

        # Remove specified columns
        chunk = chunk.drop(columns=["response", "class_index"], errors="ignore")

        # Remove duplicates within this chunk
        chunk = chunk.drop_duplicates()

        # Remove rows we've seen in previous chunks
        unique_chunk_rows = []
        for _, row in chunk.iterrows():
            row_hash = hash_row(row)
            if row_hash not in seen_hashes:
                seen_hashes.add(row_hash)
                unique_chunk_rows.append(row)

        # Convert back to DataFrame and save
        if unique_chunk_rows:
            unique_chunk = pd.DataFrame(unique_chunk_rows)

            # Write to file (append mode after first chunk)
            mode = "w" if first_chunk else "a"
            header = first_chunk
            unique_chunk.to_csv(output_path, mode=mode, header=header, index=False)

            total_rows_kept += len(unique_chunk)
            first_chunk = False

        total_rows_processed += len(chunk)

        # Progress update
        print(
            f"  Chunk {chunk_num + 1}: {len(unique_chunk_rows) if unique_chunk_rows else 0:,} unique rows kept"
        )
        print(
            f"  Total processed: {total_rows_processed:,} | Total kept: {total_rows_kept:,}"
        )

    return total_rows_kept, total_rows_processed


def main():
    parser = argparse.ArgumentParser(
        description="Reduce CSV size by removing columns and duplicates"
    )
    parser.add_argument("--data", required=True, help="Path to the input CSV file")
    parser.add_argument(
        "--output",
        "-o",
        help='Output file path (default: adds "_reduced" to input filename)',
    )
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        default=50000,
        help="Number of rows per chunk for large files (default: 50000)",
    )
    parser.add_argument(
        "--force-chunked",
        action="store_true",
        help="Force chunked processing regardless of file size",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Input file '{args.data}' not found")
        return

    # Generate output filename
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.data)
        output_path = f"{base}_reduced{ext}"

    # Check file size
    file_size_gb = get_file_size_gb(args.data)
    print(f"Input file size: {file_size_gb:.2f} GB")
    print(f"Input: {args.data}")
    print(f"Output: {output_path}")

    # Choose processing method
    use_chunked = file_size_gb >= 4.0 or args.force_chunked

    if use_chunked:
        total_kept, total_processed = process_chunked(
            args.data, output_path, args.chunksize
        )
    else:
        total_kept, total_processed = process_simple(args.data, output_path)

    print(f"\nCompleted!")
    print(f"Total rows processed: {total_processed:,}")
    print(f"Total unique rows kept: {total_kept:,}")
    print(f"Reduction: {total_processed - total_kept:,} rows removed")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
