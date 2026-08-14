# Module 1 — Data Pipeline (`/data_pipeline`)

This module scrapes book data from books.toscrape.com, cleans it, converts
the price to INR, loads it into a SQLite database, and then queries that
database using both SQL and pandas.

## What this module does, step by step

1. **Scrape** — collect raw book data from a public scraping-practice site.
2. **Clean** — fix the data types and handle any bad/missing values.
3. **Convert** — turn GBP prices into INR using a fixed rate.
4. **Load** — store everything in a proper two-table SQLite database.
5. **Query** — run SQL queries against that database.
6. **Verify** — prove that the same result can be reached using pandas alone,
   without SQL.

## How to install

```bash
pip install -r requirements.txt
```

## How to run

Run these four scripts in order. Each one depends on the file the previous
one created.

```bash
python scrape_books.py          # creates raw_books.csv
python clean_and_load.py        # creates books_clean.csv and books.db
python queries.py               # creates queries_output.txt
python pandas_verification.py   # prints a comparison, no new file
```

## What each file does

| File | What it does |
|---|---|
| `scrape_books.py` | Visits 6 pages of the site's book catalogue (about 120 books), and for each book also visits its own page to find out its category. Saves everything to `raw_books.csv`. |
| `clean_and_load.py` | Reads `raw_books.csv`, fixes the data types (price becomes a number, rating becomes an integer, stock status becomes True/False), converts price to INR, and loads everything into `books.db`. |
| `queries.py` | Runs 6 SQL queries against `books.db` and saves the results to `queries_output.txt`. |
| `pandas_verification.py` | Reads some of those same query results into pandas, and separately re-creates the JOIN query using only pandas (no SQL), to prove both approaches give the same answer. |
| `raw_books.csv` | The raw scraped data, before any cleaning. |
| `books_clean.csv` | The same data after cleaning and type conversion. |
| `books.db` | The final SQLite database. |
| `queries_output.txt` | The saved output of all 6 SQL queries. |

## Why I made these decisions

**Currency conversion.** The assignment gives a fixed rate to use:
1 GBP = 105.50 INR. This isn't a real, live exchange rate — it's a made-up
number for this project, so I didn't need to call any API or look anything
up. I just multiply every price by 105.50 to get `price_inr`.

**Why I scraped 6 whole catalogue pages instead of a few categories.**
Some categories on this site only have 1–2 books in them. If I'd picked, say,
3 specific categories, I might not have hit 60 books total. So instead I
scraped the first 6 pages of the general catalogue (about 120 books), and for
each book I visited its own page to read its category from the breadcrumb at
the top. This way I ended up with way more than 60 books, spread across many
different categories, without needing to guess which categories were big
enough.

**How I handled bad/missing data.** Two different situations came up:
- If a book's **price** couldn't be read properly, I didn't want to just
  throw away the whole row, since the title, rating, and category might still
  be fine. So instead I filled in the missing price with the **median** price
  of all the other books. This is a common, simple way to handle a missing
  number without losing the rest of the row's data.
- If a book's **rating** or **stock status** couldn't be read properly, that's
  different — these only have a few possible values (ratings are One through
  Five, stock is either "in stock" or "out of stock"), so if the text doesn't
  match any of those, something's genuinely wrong with that row. In that
  case, I just **dropped the row** instead of guessing. I print out how many
  rows got dropped so it's easy to check.

**Database design.** I made two tables that connect to each other:
```sql
categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE)
books(book_id INTEGER PRIMARY KEY, title TEXT, price_gbp REAL, price_inr REAL,
      rating INTEGER, in_stock INTEGER, category_id INTEGER REFERENCES categories(category_id))
```
Each book points to a category through `category_id`, so I'm not repeating
the category name as text in every single book row — that's what "normalized"
means here.

**The 6 SQL queries.** I made sure together they cover every clause the
assignment asked for:
1. `SELECT` / `WHERE` — in-stock books under £20
2. `ORDER BY` / `LIMIT` — the 10 most expensive books
3. `DISTINCT` — every unique category name
4. `BETWEEN` — books priced between £20 and £40
5. `IN` — books rated 4 or 5 stars
6. `JOIN` — the top 3 highest-rated books in each category

**Proving SQL and pandas agree.** For the last part, I wanted to show that
you can get the exact same JOIN result two different ways: once by asking
SQLite to do the JOIN, and once by loading the two tables into pandas and
using `pd.merge()` myself. Both end up sorted the same way (by category, then
rating, then price, all in a fixed order) so there's no ambiguity about which
rows should "win" when two books have the same rating — otherwise the two
methods could technically return the same rows but in a different order, and
look like they don't match when they actually do.

## A couple of small known quirks in the data

A few book pages on the site had slightly unusual breadcrumb text, which
caused a couple of entries to end up under odd category names like
"Add a comment" or "Default" instead of a real book genre. This doesn't
break anything — the assignment only required at least 3 real categories,
and I ended up with about 30 — but I wanted to be upfront that it's there
rather than pretend the data is perfectly clean, since real scraped data
rarely is.

## How I actually ran this

I originally wrote and tested these scripts locally in VS Code. But my
Windows machine has a security policy (Application Control) that ended up
blocking some required files — specifically Git's networking component and
NumPy's compiled code, both of which pandas depends on. Because of that, I
ran the final version of the pipeline in Google Colab instead, where those
restrictions don't apply. The code itself doesn't depend on Colab in any
way — it's ordinary Python and would run identically on any machine that has
the packages from `requirements.txt` installed.