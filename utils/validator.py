import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "psa_id",
    "domain",
    "subcategory",
    "psa_text",
}

MINIMUM_WORDS = 4
MAXIMUM_WORDS = 35


def normalize_text(text: str) -> str:
    """Normalize PSA text for duplicate checking."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


def validate_dataset(file_path: Path) -> pd.DataFrame:
    """Validate the structure and quality of a generated PSA dataset."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    dataframe["psa_text"] = dataframe["psa_text"].fillna("").astype(str)
    dataframe["normalized_text"] = dataframe["psa_text"].apply(normalize_text)
    dataframe["word_count"] = dataframe["psa_text"].str.split().str.len()

    dataframe["is_empty"] = dataframe["psa_text"].str.strip().eq("")
    dataframe["is_duplicate"] = dataframe["normalized_text"].duplicated(
        keep=False
    )
    dataframe["is_too_short"] = (
        dataframe["word_count"] < MINIMUM_WORDS
    )
    dataframe["is_too_long"] = (
        dataframe["word_count"] > MAXIMUM_WORDS
    )
    dataframe["missing_final_punctuation"] = ~dataframe[
        "psa_text"
    ].str.endswith((".", "!", "?"))

    dataframe["contains_double_space"] = dataframe[
        "psa_text"
    ].str.contains(r"\s{2,}", regex=True)

    dataframe["contains_repeated_punctuation"] = dataframe[
        "psa_text"
    ].str.contains(r"[.!?]{2,}", regex=True)

    validation_columns = [
        "is_empty",
        "is_duplicate",
        "is_too_short",
        "is_too_long",
        "missing_final_punctuation",
        "contains_double_space",
        "contains_repeated_punctuation",
    ]

    dataframe["validation_status"] = dataframe[
        validation_columns
    ].any(axis=1).map(
        {
            True: "REVIEW",
            False: "ACCEPT",
        }
    )

    return dataframe


def display_report(dataframe: pd.DataFrame) -> None:
    """Print validation statistics."""

    print("\nPSA DATASET VALIDATION REPORT")
    print("=" * 50)

    print(f"Total records: {len(dataframe):,}")
    print(
        f"Accepted records: "
        f"{(dataframe['validation_status'] == 'ACCEPT').sum():,}"
    )
    print(
        f"Records requiring review: "
        f"{(dataframe['validation_status'] == 'REVIEW').sum():,}"
    )

    print(f"Empty PSAs: {dataframe['is_empty'].sum():,}")
    print(f"Duplicate PSAs: {dataframe['is_duplicate'].sum():,}")
    print(f"Too short: {dataframe['is_too_short'].sum():,}")
    print(f"Too long: {dataframe['is_too_long'].sum():,}")
    print(
        "Missing final punctuation: "
        f"{dataframe['missing_final_punctuation'].sum():,}"
    )
    print(
        "Double spaces: "
        f"{dataframe['contains_double_space'].sum():,}"
    )
    print(
        "Repeated punctuation: "
        f"{dataframe['contains_repeated_punctuation'].sum():,}"
    )

    print("\nWord-count summary:")
    print(dataframe["word_count"].describe().round(2).to_string())

    print("\nPSAs by subcategory:")
    print(
        dataframe["subcategory"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def save_results(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save accepted and review records separately."""

    output_path.mkdir(parents=True, exist_ok=True)

    accepted = dataframe[
        dataframe["validation_status"] == "ACCEPT"
    ].copy()

    review = dataframe[
        dataframe["validation_status"] == "REVIEW"
    ].copy()

    columns_to_save = [
        "psa_id",
        "domain",
        "subcategory",
        "psa_text",
        "word_count",
        "validation_status",
    ]

    accepted[columns_to_save].to_csv(
        output_path / "health_psas_accepted.csv",
        index=False,
        encoding="utf-8",
    )

    review.to_csv(
        output_path / "health_psas_for_review.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nValidation files saved:")
    print(output_path / "health_psas_accepted.csv")
    print(output_path / "health_psas_for_review.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a generated PSA dataset."
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the generated PSA CSV file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/validated",
        help="Directory for validation results.",
    )

    args = parser.parse_args()

    dataframe = validate_dataset(Path(args.dataset))

    display_report(dataframe)

    save_results(
        dataframe=dataframe,
        output_path=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()