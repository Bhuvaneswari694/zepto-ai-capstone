"""
queries.py
----------

Module 1 - Data Pipeline (Zepto AI/ML Capstone)

This script executes the required SQL queries against the SQLite database.

The assignment requires the project to demonstrate the following SQL
operations:

    - SELECT
    - WHERE
    - ORDER BY
    - LIMIT
    - DISTINCT
    - IN or BETWEEN
    - JOIN

Each query is executed against books.db, and both the SQL statement and
its output are displayed in the console.

The results are also saved to queries_output.txt so they can be included
in the final project submission.

Run:

    python queries.py

Output:

    queries_output.txt
"""

import sqlite3
import pandas as pd


DB_PATH = "books.db"


# Each query demonstrates one or more SQL concepts required by the
# assignment.

QUERIES = [

    (
        "Q1: SELECT / WHERE -- In-stock books priced under 20 GBP",

        """
        SELECT title, price_gbp, in_stock
        FROM books
        WHERE in_stock = 1
        AND price_gbp < 20
        """
    ),

    (
        "Q2: ORDER BY / LIMIT -- 10 most expensive books (INR)",

        """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
        """
    ),

    (
        "Q3: DISTINCT -- Unique categories",

        """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
        """
    ),

    (
        "Q4: BETWEEN -- Books priced between 20 and 40 GBP",

        """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
        """
    ),

    (
        "Q5: IN -- Books with ratings of 4 or 5 stars",

        """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, title
        """
    ),

    (
        "Q6: JOIN -- Top 3 highest-rated books from each category",

        """
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
    ),

]


def main():
    """
    Connect to the database, execute all queries, display the results,
    and save the output to a text file.
    """

    conn = sqlite3.connect(DB_PATH)

    output_lines = []

    for title, sql in QUERIES:

        # Read each query directly into a Pandas DataFrame.
        # This makes the output easier to format and display.

        df = pd.read_sql(sql, conn)

        block = (
            f"\n{'=' * 70}\n"
            f"{title}\n"
            f"SQL:\n"
            f"{sql.strip()}\n\n"
            f"Output ({len(df)} rows):\n"
            f"{df.to_string(index=False)}\n"
        )

        print(block)

        output_lines.append(block)

    # Save all query results to a separate file so the output can be
    # reviewed without running the script again.

    with open(
        "queries_output.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(output_lines)
        )

    conn.close()

    print(
        "\nSaved all query results to queries_output.txt"
    )


if __name__ == "__main__":
    main()