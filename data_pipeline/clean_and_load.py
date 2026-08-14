"""
clean_and_load.py
-----------------

Module 1 - Data Pipeline (Zepto AI/ML Capstone)

This script reads the raw data collected by scrape_books.py, cleans and
converts the scraped fields into appropriate data types, and loads the
processed data into a normalized SQLite database.

The assignment requires the following transformations:

    - Convert price from text to float
    - Convert star ratings from text to integers
    - Convert availability into Boolean values
    - Convert GBP to INR using the project's fixed conversion rate

The cleaned data is saved as books_clean.csv.

The final dataset is then loaded into a two-table SQLite database that
uses a primary key and foreign key relationship between categories and
books.

Run:

    python clean_and_load.py

Output:

    books_clean.csv
    books.db
"""

import re
import sqlite3
import pandas as pd


RAW_CSV = "raw_books.csv"
CLEAN_CSV = "books_clean.csv"
DB_PATH = "books.db"


# The assignment requires a fixed conversion rate.
# This is a project-defined constant and doesn't require an API call.

GBP_TO_INR = 105.50


# Convert text-based ratings into numeric values.

RATING_WORD_TO_INT = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def parse_price(price_text):
    """
    Convert a price such as '£51.77' into a floating-point value.

    If the price can't be parsed, return None and allow the cleaning
    function to decide how the missing value should be handled.
    """

    if price_text is None:
        return None

    match = re.search(r"[\d.]+", str(price_text))

    if match:
        return float(match.group())

    return None


def parse_rating(star_rating_text):
    """
    Convert text-based ratings into integers.

    Example:

        'Three' -> 3

    Return None if the rating doesn't match any expected value.
    """

    return RATING_WORD_TO_INT.get(
        str(star_rating_text).strip()
    )


def parse_availability(availability_text):
    """
    Convert availability text into Boolean values.

    Examples:

        'In stock' -> True

        'Out of stock' -> False
    """

    if availability_text is None:
        return None

    return "in stock" in str(
        availability_text
    ).lower()


def clean_dataframe(df):
    """
    Clean all scraped fields and convert them into the required data types.

    The assignment requires:

        price_gbp -> float

        rating -> integer

        in_stock -> Boolean

        price_inr -> float
    """

    df = df.copy()

    df["price_gbp"] = df["price"].apply(
        parse_price
    )

    df["rating"] = df["star_rating"].apply(
        parse_rating
    )

    df["in_stock"] = df["availability"].apply(
        parse_availability
    )

    rows_before = len(df)

    # If a price can't be parsed, use median imputation instead of
    # removing the entire record.

    if df["price_gbp"].isna().any():

        median_price = df["price_gbp"].median()

        df["price_gbp"] = df["price_gbp"].fillna(
            median_price
        )

        print(
            f"Median price used for missing values: "
            f"{median_price:.2f}"
        )

    # Unexpected values in rating or availability are treated as
    # invalid records and are removed.

    invalid_rows = (
        df["rating"].isna()
        | df["in_stock"].isna()
    )

    if invalid_rows.any():

        print(
            f"Dropping {invalid_rows.sum()} invalid row(s)."
        )

    df = df[~invalid_rows].copy()

    rows_after = len(df)

    print(
        f"Rows before cleaning: {rows_before}"
    )

    print(
        f"Rows after cleaning: {rows_after}"
    )

    df["rating"] = df["rating"].astype(
        int
    )

    df["in_stock"] = df["in_stock"].astype(
        bool
    )

    # Required currency conversion.

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    df["category"] = (
        df["category"]
        .astype(str)
        .str.strip()
    )

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
        ]
    ]


def build_database(df):
    """
    Create a normalized SQLite database.

    Two tables are created:

        categories

        books

    The tables are connected through a primary key and foreign key
    relationship.
    """

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS books;

        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories
        (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,

            category_name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE books
        (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,

            title TEXT NOT NULL,

            price_gbp REAL NOT NULL,

            price_inr REAL NOT NULL,

            rating INTEGER NOT NULL,

            in_stock INTEGER NOT NULL,

            category_id INTEGER NOT NULL,

            FOREIGN KEY (category_id)
            REFERENCES categories(category_id)
        );
        """
    )

    # Insert categories first because the books table depends on them.

    categories = sorted(
        df["category"].unique()
    )

    cursor.executemany(
        """
        INSERT INTO categories
        (category_name)
        VALUES (?)
        """,
        [(category,) for category in categories],
    )

    conn.commit()

    # Create a mapping between category names and category IDs.

    category_map = dict(
        cursor.execute(
            """
            SELECT
                category_name,
                category_id
            FROM categories
            """
        ).fetchall()
    )

    # Replace category names with their corresponding IDs before
    # inserting records into the books table.

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            int(row.rating),
            int(row.in_stock),
            category_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]

    cursor.executemany(
        """
        INSERT INTO books
        (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        )
        VALUES
        (?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )

    conn.commit()

    conn.close()

    print(
        f"Loaded {len(categories)} categories and "
        f"{len(book_rows)} books into {DB_PATH}"
    )


def main():
    """
    Run the complete cleaning and loading workflow.

    1. Read the raw CSV file.
    2. Clean and transform the data.
    3. Save the cleaned dataset.
    4. Create the SQLite database.
    """

    raw_df = pd.read_csv(
        RAW_CSV
    )

    clean_df = clean_dataframe(
        raw_df
    )

    clean_df.to_csv(
        CLEAN_CSV,
        index=False,
    )

    print(
        f"Saved cleaned data to {CLEAN_CSV}"
    )

    build_database(
        clean_df
    )


if __name__ == "__main__":
    main()