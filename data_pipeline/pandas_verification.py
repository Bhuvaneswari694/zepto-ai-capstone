"""
pandas_verification.py
----------------------

Module 1 - Data Pipeline (Zepto AI/ML Capstone)

This script verifies the final database by completing two tasks required
by the assignment.

1. Read SQL query results directly into Pandas using pd.read_sql().

2. Reproduce the JOIN query using pd.merge() instead of SQL and verify
   that both approaches produce the same result.

This comparison demonstrates that SQL joins and Pandas merges can be
used to combine related tables and retrieve equivalent information.

Run:

    python pandas_verification.py
"""

import sqlite3
import pandas as pd

DB_PATH = "books.db"


def main():
    """
    Execute all verification steps required by the assignment.

    1. Load SQL query results into Pandas.

    2. Recreate the JOIN operation using only Pandas.

    3. Compare both results and confirm that they are equivalent.
    """

    conn = sqlite3.connect(DB_PATH)

    # ------------------------------------------------------------------
    # Part 1: Read SQL query results directly into Pandas.
    # ------------------------------------------------------------------

    print("=" * 70)
    print("pd.read_sql(): In-stock books priced under 20 GBP")

    df_read_sql_1 = pd.read_sql(
        """
        SELECT
            title,
            price_gbp,
            in_stock
        FROM books
        WHERE in_stock = 1
        AND price_gbp < 20
        """,
        conn,
    )

    print(df_read_sql_1.to_string(index=False))

    print("\n" + "=" * 70)
    print("pd.read_sql(): Distinct category names")

    df_read_sql_2 = pd.read_sql(
        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
        """,
        conn,
    )

    print(df_read_sql_2.to_string(index=False))

    # ------------------------------------------------------------------
    # Part 2: Retrieve the JOIN result using SQL.
    # ------------------------------------------------------------------

    sql_join_query = """
        SELECT
            c.category_name,
            b.title,
            b.rating,
            b.price_inr
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        WHERE b.book_id IN
        (
            SELECT b2.book_id
            FROM books b2
            WHERE b2.category_id = b.category_id
            ORDER BY
                b2.rating DESC,
                b2.price_inr DESC
            LIMIT 3
        )
        ORDER BY
            c.category_name,
            b.rating DESC,
            b.price_inr DESC
    """

    df_sql_join = pd.read_sql(
        sql_join_query,
        conn,
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Part 3: Reproduce the same JOIN operation using only Pandas.
    # ------------------------------------------------------------------

    books_df = pd.read_sql(
        "SELECT * FROM books",
        conn,
    )

    categories_df = pd.read_sql(
        "SELECT * FROM categories",
        conn,
    )

    # Merge the two tables using the shared category_id column.

    merged = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner",
    )

    # Apply the same sorting logic used in the SQL query.

    merged_sorted = merged.sort_values(
        [
            "category_name",
            "rating",
            "price_inr",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    # Keep only the top three books from each category.

    df_pandas_join = (
        merged_sorted
        .groupby(
            "category_name",
            group_keys=False,
        )
        .head(3)[
            [
                "category_name",
                "title",
                "rating",
                "price_inr",
            ]
        ]
        .sort_values(
            [
                "category_name",
                "rating",
                "price_inr",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Display both results.
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SQL JOIN result (pd.read_sql())")

    print(df_sql_join.to_string(index=False))

    print("\n" + "=" * 70)
    print("Equivalent result using pd.merge()")

    print(df_pandas_join.to_string(index=False))

    # ------------------------------------------------------------------
    # Verify that both approaches return identical results.
    # ------------------------------------------------------------------

    are_equal = df_sql_join.equals(
        df_pandas_join
    )

    print("\n" + "=" * 70)

    print(
        f"SQL JOIN and Pandas merge produce the same result: "
        f"{are_equal}"
    )

    conn.close()


if __name__ == "__main__":
    main()