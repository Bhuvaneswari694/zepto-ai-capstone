"""
scrape_books.py
----------------

Module 1 - Data Pipeline (Zepto AI/ML Capstone)

This script collects raw book data from books.toscrape.com and saves it
to a CSV file.

I decided to scrape the first six catalogue pages instead of selecting
individual categories. This guarantees that the dataset contains more
than the required minimum of 60 books.

The catalogue pages provide most of the required fields:

    - Title
    - Price
    - Star rating
    - Availability

Category information isn't available on the listing pages, so the script
opens each book's product page and extracts the category from the
breadcrumb navigation.

Output:

    raw_books.csv

Required libraries:

    pip install requests beautifulsoup4

Run:

    python scrape_books.py
"""

import csv
import time

import requests
from bs4 import BeautifulSoup


# Website configuration
BASE_URL = "https://books.toscrape.com/"
CATALOGUE_PAGE = BASE_URL + "catalogue/page-{}.html"

# Six pages provide far more than the required 60 books.
NUM_PAGES = 6

# Add a small delay between requests.
# Even though this website supports scraping, it's still good practice
# to avoid sending requests too quickly.
REQUEST_DELAY = 0.3


def get_soup(url):
    """
    Send an HTTP request and convert the response into a BeautifulSoup
    object.

    Any HTTP errors are raised so they can be handled by the calling
    function.
    """

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_category_for_book(book_url):
    """
    Retrieve a book's category from its product page.

    The listing pages don't include category information, so each book's
    page must be visited individually.

    If the request fails, return 'Unknown' and continue processing the
    remaining books.
    """

    try:
        soup = get_soup(book_url)

        breadcrumb = soup.select("ul.breadcrumb li a")

        # Example:
        # Home > Books > Travel > Book Title
        # The actual category is stored at index 2.

        if len(breadcrumb) >= 3:
            return breadcrumb[2].get_text(strip=True)

    except requests.RequestException as error:

        print(f"Could not retrieve the category: {error}")

    return "Unknown"


def scrape_all_books():
    """
    Scrape all selected catalogue pages and extract the required fields.

    Returns:

        A list of dictionaries containing:

        - title
        - price
        - star_rating
        - availability
        - category
    """

    rows = []

    for page_number in range(1, NUM_PAGES + 1):

        page_url = CATALOGUE_PAGE.format(page_number)

        print(f"Scraping page {page_number}: {page_url}")

        try:
            # If one page fails to load, skip it and continue with the
            # remaining pages.

            soup = get_soup(page_url)

        except requests.RequestException as error:

            print(f"Skipping page {page_number}: {error}")

            continue

        articles = soup.select("article.product_pod")

        for article in articles:

            try:
                # Extract all required information from a single book card.

                title = article.h3.a["title"].strip()

                price_text = article.select_one(
                    "p.price_color"
                ).get_text(strip=True)

                # Example:
                # class="star-rating Three"
                # Remove "star-rating" and keep "Three".

                rating_classes = article.select_one(
                    "p.star-rating"
                )["class"]

                star_rating = [
                    item
                    for item in rating_classes
                    if item != "star-rating"
                ][0]

                availability = article.select_one(
                    "p.instock.availability"
                ).get_text(strip=True)

                relative_link = article.h3.a["href"]

                book_url = (
                    BASE_URL
                    + "catalogue/"
                    + relative_link.replace("../../../", "")
                )

            except (
                AttributeError,
                KeyError,
                IndexError,
                TypeError,
            ) as error:

                # Skip malformed records.
                # One bad book shouldn't stop the entire pipeline.

                print(f"Skipping an invalid product card: {error}")

                continue

            category = get_category_for_book(book_url)

            rows.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": category,
                }
            )

            time.sleep(REQUEST_DELAY)

        time.sleep(REQUEST_DELAY)

    return rows


def main():
    """
    Run the complete scraping workflow.

    1. Scrape book data.
    2. Verify that at least 60 books were collected.
    3. Save the results to a CSV file.
    """

    rows = scrape_all_books()

    print(f"\nTotal books scraped: {len(rows)}")

    categories_found = {
        row["category"]
        for row in rows
    }

    print(
        f"Categories discovered ({len(categories_found)}): "
        f"{sorted(categories_found)}"
    )

    if len(rows) < 60:

        print(
            "Warning: Fewer than 60 books were collected. "
            "Increase NUM_PAGES and run the script again."
        )

    with open(
        "raw_books.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "title",
                "price",
                "star_rating",
                "availability",
                "category",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Saved raw_books.csv")


if __name__ == "__main__":
    main()