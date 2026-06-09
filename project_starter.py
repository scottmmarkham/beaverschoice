import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import re
from typing import Dict, List, Union, Any
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }

def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

dotenv.load_dotenv()

API_KEY = os.getenv("UDACITY_OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing UDACITY_OPENAI_API_KEY in environment.")

if API_KEY.startswith("voc-"):
    os.environ["OPENAI_BASE_URL"] = "https://openai.vocareum.com/v1"

model_kwargs = {
    "model_id": "gpt-4o-mini",
    "api_key": API_KEY,
}

model = OpenAIServerModel(**model_kwargs)

PRICE_LIST = {item["item_name"]: float(item["unit_price"]) for item in paper_supplies}
SUPPORTED_ITEMS = set(PRICE_LIST.keys())

ITEM_ALIASES = {
    "streamers": "Party streamers",
    "party streamers": "Party streamers",
    "a4": "A4 paper",
    "a4 paper": "A4 paper",
    "printer paper": "A4 paper",
    "printing paper": "A4 paper",
    "a4 printer paper": "A4 paper",
    "a4 printing paper": "A4 paper",
    "letter paper": "Letter-sized paper",
    "letter-sized paper": "Letter-sized paper",
    "letter sized paper": "Letter-sized paper",
    "cups": "Paper cups",
    "paper cups": "Paper cups",
    "plates": "Paper plates",
    "paper plates": "Paper plates",
    "napkins": "Paper napkins",
    "paper napkins": "Paper napkins",
    "poster paper": "Poster paper",
    "banner paper": "Banner paper",
    "card stock": "Cardstock",
    "cardstock": "Cardstock",
    "cardstock paper": "Cardstock",
    "paper napkin": "Paper napkins",
    "paper plate": "Paper plates",
    "paper cup": "Paper cups",
    "plate": "Paper plates",
    "cup": "Paper cups",
    "napkin": "Paper napkins",
    "table cover": "Table covers",
    "table covers": "Table covers",
    "flyer": "Flyers",
    "flyers": "Flyers",
    "notepad": "Notepads",
    "notepads": "Notepads",
    "envelope": "Envelopes",
    "envelopes": "Envelopes",
    "sticky note": "Sticky notes",
    "sticky notes": "Sticky notes",
    "invitation card": "Invitation cards",
    "invitation cards": "Invitation cards",
    "folder": "Presentation folders",
    "folders": "Presentation folders",
    "poster board": "Large poster paper (24x36 inches)",
    "display board": "Large poster paper (24x36 inches)",
}


def normalize_item_name(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^\w\s\-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in ITEM_ALIASES:
        return ITEM_ALIASES[cleaned]

    for supported in SUPPORTED_ITEMS:
        if cleaned == supported.lower():
            return supported

    # phrase-based normalization for common descriptive requests
    if "a4" in cleaned and ("paper" in cleaned or "printer" in cleaned or "printing" in cleaned):
        return "A4 paper"

    if ("letter" in cleaned or "letter-sized" in cleaned) and "paper" in cleaned:
        return "Letter-sized paper"

    if "cardstock" in cleaned or "card stock" in cleaned:
        return "Cardstock"

    if "colored paper" in cleaned or "colour paper" in cleaned:
        return "Colored paper"

    if "glossy" in cleaned and "paper" in cleaned:
        return "Glossy paper"

    if "matte" in cleaned and "paper" in cleaned:
        return "Matte paper"

    if "recycled" in cleaned and "paper" in cleaned:
        return "Recycled paper"

    if "eco-friendly" in cleaned and "paper" in cleaned:
        return "Eco-friendly paper"

    if "poster" in cleaned and "paper" in cleaned:
        return "Poster paper"

    if "banner" in cleaned and "paper" in cleaned:
        return "Banner paper"

    if "kraft" in cleaned and "paper" in cleaned:
        return "Kraft paper"

    if "construction" in cleaned and "paper" in cleaned:
        return "Construction paper"

    if "wrapping" in cleaned and "paper" in cleaned:
        return "Wrapping paper"

    if "glitter" in cleaned and "paper" in cleaned:
        return "Glitter paper"

    if "decorative" in cleaned and "paper" in cleaned:
        return "Decorative paper"

    if "letterhead" in cleaned and "paper" in cleaned:
        return "Letterhead paper"

    if "legal" in cleaned and "paper" in cleaned:
        return "Legal-size paper"

    if "crepe" in cleaned and "paper" in cleaned:
        return "Crepe paper"

    if "photo" in cleaned and "paper" in cleaned:
        return "Photo paper"

    if "uncoated" in cleaned and "paper" in cleaned:
        return "Uncoated paper"

    if "butcher" in cleaned and "paper" in cleaned:
        return "Butcher paper"

    if "heavyweight" in cleaned and "paper" in cleaned:
        return "Heavyweight paper"

    if "standard copy paper" in cleaned or ("standard" in cleaned and "copy paper" in cleaned):
        return "Standard copy paper"

    if "bright-colored" in cleaned and "paper" in cleaned:
        return "Bright-colored paper"

    if "patterned" in cleaned and "paper" in cleaned:
        return "Patterned paper"

    if "plate" in cleaned:
        return "Paper plates"

    if "cup" in cleaned:
        return "Paper cups"

    if "napkin" in cleaned:
        return "Paper napkins"

    if "disposable cup" in cleaned:
        return "Disposable cups"

    if "table cover" in cleaned:
        return "Table covers"

    if "envelope" in cleaned:
        return "Envelopes"

    if "sticky note" in cleaned:
        return "Sticky notes"

    if "notepad" in cleaned:
        return "Notepads"

    if "invitation card" in cleaned:
        return "Invitation cards"

    if "flyer" in cleaned:
        return "Flyers"

    if "streamer" in cleaned:
        return "Party streamers"

    if "washi tape" in cleaned or "decorative adhesive tape" in cleaned:
        return "Decorative adhesive tape (washi tape)"

    if "party bag" in cleaned:
        return "Paper party bags"

    if "name tag" in cleaned:
        return "Name tags with lanyards"

    if "presentation folder" in cleaned or "folder" in cleaned:
        return "Presentation folders"

    if "large poster" in cleaned or "poster board" in cleaned or "display board" in cleaned:
        return "Large poster paper (24x36 inches)"

    if "banner roll" in cleaned or "rolls of banner paper" in cleaned:
        return "Rolls of banner paper (36-inch width)"

    if "100 lb cover stock" in cleaned or "cover stock" in cleaned:
        return "100 lb cover stock"

    if "80 lb text paper" in cleaned or "text paper" in cleaned:
        return "80 lb text paper"

    if "250 gsm cardstock" in cleaned or "250gsm cardstock" in cleaned:
        return "250 gsm cardstock"

    if "220 gsm poster paper" in cleaned or "220gsm poster paper" in cleaned:
        return "220 gsm poster paper"

    return name.strip()

def parse_agent_dict_output(agent_output: Any) -> Dict[str, Any]:
    text = str(agent_output).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
        return {"raw_output": text, "_parse_failed": True}
    except Exception:
        return {"raw_output": text, "_parse_failed": True}

def extract_stock_number(agent_output: Any) -> int:
    text = str(agent_output)
    match = re.search(r"current_stock\s*=\s*(\d+)", text)
    if match:
        return int(match.group(1))
    return 0

def extract_delivery_date(agent_output: Any) -> str:
    text = str(agent_output)
    match = re.search(r"delivery_date\s*=\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    if match:
        return match.group(1)

    fallback = re.search(r"([0-9]{4}-[0-9]{2}-[0-9]{2})", text)
    if fallback:
        return fallback.group(1)

    return ""

def calculate_discount(subtotal: float) -> float:
    if subtotal > 500:
        return 0.10
    if subtotal > 200:
        return 0.05
    return 0.0

def parse_customer_request(request_text: str) -> Dict[str, Any]:
    requested_date = "2025-01-15"
    date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", request_text)
    if date_match:
        requested_date = date_match.group(1)

    text = request_text

    # Remove parenthetical request-date wrapper
    text = re.sub(r"\(.*?Date of request:.*?\)", "", text, flags=re.IGNORECASE)

    # Normalize punctuation and whitespace
    text = text.replace(";", ",")
    text = re.sub(r"[!?]", ".", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Split into candidate chunks
    chunks = re.split(r",|\band\b|\.", text, flags=re.IGNORECASE)

    items = []
    unsupported_items = []
    merged_items = {}


    filler_prefix = (
        r"^(please|pls|i need|i want|we need|we want|can i order|"
        r"i'd like|i would like|order|need|send|ship)\s+"
    )

    trailing_filler = r"\s+(for|by|on|before|after)\b.*$"

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        chunk = re.sub(filler_prefix, "", chunk, flags=re.IGNORECASE).strip()

        # Require quantity followed by a candidate item phrase
        match = re.search(r"(\d+)\s+([A-Za-z0-9\-\(\) ]+)", chunk)
        if not match:
            continue

        quantity = int(match.group(1))
        raw_name = match.group(2).strip()

        # Trim trailing scheduling/context filler
        raw_name = re.sub(trailing_filler, "", raw_name, flags=re.IGNORECASE).strip()

        # Trim generic trailing words that often appear after item names
        raw_name = re.sub(r"\b(please|thanks|thank you)$", "", raw_name, flags=re.IGNORECASE).strip()

        normalized_name = normalize_item_name(raw_name)

        if normalized_name in SUPPORTED_ITEMS:
            if normalized_name in merged_items:
                merged_items[normalized_name]["quantity"] += quantity
            else:
                merged_items[normalized_name] = {
                    "raw_name": raw_name,
                    "item_name": normalized_name,
                    "quantity": quantity,
                }

        else:
            unsupported_items.append({
                "raw_name": raw_name,
                "normalized_name": normalized_name,
                "quantity": quantity,
            })

    items = list(merged_items.values())

    print("PARSED ITEMS:", items)
    print("UNSUPPORTED ITEMS:", unsupported_items)

    return {
        "requested_date": requested_date,
        "items": items,
        "unsupported_items": unsupported_items,
    }

def extract_json_dict(raw_output: Any) -> Dict[str, Any]:
    """
    Best-effort extraction of a JSON object from orchestrator output.
    """
    if isinstance(raw_output, dict):
        return raw_output

    if raw_output is None:
        return {}

    text = str(raw_output).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def normalize_orchestrator_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply light validation/defaulting so downstream logic is deterministic.
    """
    normalized = {
        "customer_response": result.get(
            "customer_response",
            "We could not fully process your request."
        ),
        "stock_status": result.get("stock_status", "failed"),
        "quoted_total": float(result.get("quoted_total", 0.0) or 0.0),
        "estimated_delivery": result.get("estimated_delivery"),
        "fulfilled_items": result.get("fulfilled_items", []) or [],
        "restock_orders": result.get("restock_orders", []) or [],
        "unsupported_items": result.get("unsupported_items", []) or [],
    }

    cleaned_unsupported = []
    for item in normalized["unsupported_items"]:
        if isinstance(item, str):
            name = item.strip()
            if name:
                cleaned_unsupported.append(name)
        elif isinstance(item, dict):
            name = item.get("item_name") or item.get("name") or ""
            name = str(name).strip()
            if name:
                cleaned_unsupported.append(name)

    normalized["unsupported_items"] = cleaned_unsupported
    return normalized

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# Tools for inventory agent
@tool
def inventory_check_tool(item_name: str, as_of_date: str) -> str:
    """
    Check the current inventory level for a single item as of a given date.

    Args:
        item_name: The normalized inventory item name to look up.
        as_of_date: The date to check inventory against, in YYYY-MM-DD format.

    Returns:
        A compact string containing the item name and current stock level.
    """
    stock_df = get_stock_level(item_name, as_of_date)

    if stock_df is None or stock_df.empty or "current_stock" not in stock_df.columns:
        current_stock = 0
    else:
        current_stock = int(stock_df["current_stock"].iloc[0] or 0)

    return str({
        "item_name": item_name,
        "current_stock": current_stock
    })

@tool
def supplier_delivery_tool(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date for a requested quantity ordered on a given date.

    Args:
        input_date_str: The order date in YYYY-MM-DD format.
        quantity: The number of units that need to be restocked.

    Returns:
        A compact string containing the estimated supplier delivery date.
    """
    delivery_date = get_supplier_delivery_date(input_date_str, quantity)
    return str({
        "delivery_date": delivery_date
    })


# Tools for quoting agent
@tool
def quote_history_tool(search_terms: str) -> str:
    """
    Search recent quote history using one or more comma-separated search terms.

    Args:
        search_terms: A comma-separated string of keywords to search for in quote history.

    Returns:
        A string representation of up to five matching quote history results.
    """
    terms = [t.strip() for t in search_terms.split(",") if t.strip()]
    results = search_quote_history(terms, limit=5)
    return str(results)

# Tools for ordering agent
@tool
def create_sales_transaction_tool(item_name: str, quantity: int, price: float, date: str) -> str:
    """
    Create a sales transaction record for an item.

    Args:
        item_name: The normalized inventory item name being sold.
        quantity: The number of units sold.
        price: The total price for the transaction.
        date: The transaction date in YYYY-MM-DD format.

    Returns:
        A compact string containing the created sales transaction identifier.
    """
    transaction_id = create_transaction(item_name, "sales", quantity, price, date)
    return str({
        "sales_transaction_id": transaction_id
    })


@tool
def create_stock_order_transaction_tool(item_name: str, quantity: int, price: float, date: str) -> str:
    """
    Create a stock order transaction record for an item being restocked.

    Args:
        item_name: The normalized inventory item name being ordered.
        quantity: The number of units being ordered from the supplier.
        price: The total price for the stock order transaction.
        date: The transaction date in YYYY-MM-DD format.

    Returns:
        A compact string containing the created stock order transaction identifier.
    """
    transaction_id = create_transaction(item_name, "stock_orders", quantity, price, date)
    return str({
        "stock_order_transaction_id": transaction_id
    })
# Set up your agents and create an orchestration agent that will manage them.

inventory_agent = CodeAgent(
    tools=[inventory_check_tool, supplier_delivery_tool],
    model=model,
    name="inventory_agent",
    description="Use tools to check stock levels and supplier delivery dates. Return structured dictionary-style outputs only.",
)

quote_agent = CodeAgent(
    tools=[quote_history_tool],
    model=model,
    name="quote_agent",
    description="Use quote history tool to retrieve similar historical quotes. Return structured dictionary-style outputs only.",
)

order_agent = CodeAgent(
    tools=[create_sales_transaction_tool, create_stock_order_transaction_tool],
    model=model,
    name="order_agent",
    description="Use transaction tools to create sales and stock-order records. Return structured dictionary-style outputs only.",
)

orchestrator_agent = ToolCallingAgent(
    tools=[],
    managed_agents=[inventory_agent, quote_agent, order_agent],
    model=model,
    name="orchestrator_agent",
    description=(
        "Primary workflow controller. Delegates to worker agents, decides fulfillment/restocking flow, "
        "and returns one final structured JSON plan. The Python layer persists transactions deterministically "
        "after the orchestrator returns its result."
    ),
)

def build_orchestrator_prompt(user_request: str, parsed_request: Dict[str, Any]) -> str:
    requested_date = parsed_request.get("requested_date", datetime.now().strftime("%Y-%m-%d"))
    items = parsed_request.get("items", [])

    return f"""
You are the orchestration agent for Beaver's Choice Paper Company.

Your job:
1. Understand the customer request.
2. Delegate to the inventory, quote, and order agents as needed.
3. Decide whether the request can be fulfilled now, needs restocking, is unsupported, or failed.
4. Return ONE final JSON object only.
5. Do NOT record transactions yourself. The Python application will persist transactions after your response.

Customer request:
{user_request}

Parsed request date:
{requested_date}

Parsed items:
{json.dumps(items, indent=2)}

You must return valid JSON with exactly this schema:
{{
  "customer_response": "string",
  "stock_status": "ready_to_fulfill | pending_restock | unsupported | failed",
  "quoted_total": 0.0,
  "estimated_delivery": "string",
  "fulfilled_items": [
    {{
      "item_name": "string",
      "quantity": 0,
      "unit_price": 0.0
    }}
  ],
  "restock_orders": [
    {{
      "item_name": "string",
      "quantity": 0,
      "unit_cost": 0.0
    }}
  ],
  "unsupported_items": ["string"]
}}

Rules:
- Return JSON only. No markdown. No explanation outside the JSON.
- If items can be sold now, put them in fulfilled_items and use stock_status="ready_to_fulfill".
- If items require purchasing inventory first, put them in restock_orders and use stock_status="pending_restock".
- If some items are not sold by the company, list them in unsupported_items and use stock_status="unsupported" unless other supported items are still being processed.
- quoted_total should reflect the customer-facing total quote when applicable.
- unit_price is the customer sale price per unit.
- unit_cost is the company purchase cost per unit for restocking.
- If something goes wrong, return stock_status="failed" with the best possible customer_response.
""".strip()

def commit_transactions_from_result(
    parsed_request: Dict[str, Any],
    result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Persist transactions deterministically in Python based on the orchestrator result.
    Uses the exact parsed request date so financial reports for that date reflect the writes.
    Includes duplicate protection to avoid double-recording the same transaction.
    """
    committed = []

    requested_date = parsed_request.get("requested_date", datetime.now().strftime("%Y-%m-%d"))
    fulfilled_items = result.get("fulfilled_items", []) or []
    restock_orders = result.get("restock_orders", []) or []

    existing_tx = get_transactions_for_date(requested_date)

    def already_recorded(item_name: str, transaction_type: str, quantity: int, total_price: float) -> bool:
        if existing_tx.empty:
            return False

        matches = existing_tx[
            (existing_tx["item_name"] == item_name) &
            (existing_tx["transaction_type"] == transaction_type) &
            (existing_tx["units"] == quantity) &
            (existing_tx["price"].round(2) == round(float(total_price), 2))
        ]
        return not matches.empty

    for item in fulfilled_items:
        item_name = item.get("item_name")
        quantity = int(item.get("quantity", 0) or 0)
        unit_price = float(item.get("unit_price", 0.0) or 0.0)

        if not item_name or quantity <= 0:
            continue

        total_price = quantity * unit_price

        if not already_recorded(item_name, "sales", quantity, total_price):
            create_transaction(
                item_name=item_name,
                quantity=quantity,
                transaction_type="sales",
                price=total_price,
                date=requested_date,
            )
            committed.append(
                {
                    "item_name": item_name,
                    "transaction_type": "sales",
                    "quantity": quantity,
                    "price": total_price,
                    "date": requested_date,
                }
            )

    for item in restock_orders:
        item_name = item.get("item_name")
        quantity = int(item.get("quantity", 0) or 0)
        unit_cost = float(item.get("unit_cost", 0.0) or 0.0)

        if not item_name or quantity <= 0:
            continue

        total_cost = quantity * unit_cost

        if not already_recorded(item_name, "stock_orders", quantity, total_cost):
            create_transaction(
                item_name=item_name,
                quantity=quantity,
                transaction_type="stock_orders",
                price=total_cost,
                date=requested_date,
            )
            committed.append(
                {
                    "item_name": item_name,
                    "transaction_type": "stock_orders",
                    "quantity": quantity,
                    "price": total_cost,
                    "date": requested_date,
                }
            )

    return committed

def call_your_multi_agent_system(user_request: str) -> Dict[str, Any]:
    """
    Parse the request, run the orchestrator, normalize its JSON result,
    commit transactions deterministically, and return customer-facing output.
    """
    parsed_request = parse_customer_request(user_request)
    prompt = build_orchestrator_prompt(user_request, parsed_request)

    raw_response = orchestrator_agent.run(prompt)
    raw_text = str(raw_response)

    extracted = extract_json_dict(raw_text)
    normalized = normalize_orchestrator_result(extracted)

    committed = commit_transactions_from_result(parsed_request, normalized)

    return {
        "customer_response": normalized.get(
            "customer_response",
            "We could not fully process your request."
        ),
        "stock_status": normalized.get("stock_status", "failed"),
        "quoted_total": normalized.get("quoted_total", 0.0),
        "estimated_delivery": normalized.get("estimated_delivery", ""),
        "fulfilled_items": normalized.get("fulfilled_items", []),
        "restock_orders": normalized.get("restock_orders", []),
        "unsupported_items": normalized.get("unsupported_items", []),
        "transactions_committed": committed,
        "internal_result": normalized,
    }

def get_transactions_for_date(request_date: str) -> pd.DataFrame:
    """
    Return all transactions recorded for a specific request date.
    """
    return pd.read_sql(
        """
        SELECT item_name, transaction_type, units, price, transaction_date
        FROM transactions
        WHERE transaction_date = :request_date
        ORDER BY item_name, transaction_type
        """,
        db_engine,
        params={"request_date": request_date},
    )

def audit_request_transactions(request_date: str, response: Dict[str, Any]) -> pd.DataFrame:
    """
    Inspect transactions created for the current request date and warn if the
    response suggests fulfilment/restocking but no matching transactions exist.
    """
    tx = get_transactions_for_date(request_date)

    stock_status = response.get("stock_status", "")
    quoted_total = response.get("quoted_total", 0.0)

    if stock_status in {"ready_to_fulfill", "pending_restock"} and tx.empty:
        print(
            f"WARNING: Request on {request_date} returned stock_status='{stock_status}' "
            f"but no transactions were recorded for that date."
        )

    if stock_status == "ready_to_fulfill":
        sales_count = (tx["transaction_type"] == "sales").sum() if not tx.empty else 0
        if sales_count == 0:
            print(
                f"WARNING: Request on {request_date} appears fulfilled "
                f"(quoted_total={quoted_total}) but no sales transaction was recorded."
            )

    if stock_status == "pending_restock" and not tx.empty:
        sales_count = (tx["transaction_type"] == "sales").sum()
        stock_order_count = (tx["transaction_type"] == "stock_orders").sum()
        if sales_count == 0 and stock_order_count == 0:
            print(
                f"WARNING: Request on {request_date} is pending_restock but no sales or stock_orders "
                f"transactions were recorded."
            )

    return tx


def run_test_scenarios():
    print("Initializing Database...")
    init_database(db_engine)

    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"],
            format="%m/%d/%y",
            errors="coerce",
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []

    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx + 1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"

        response = call_your_multi_agent_system(request_with_date)

        print("Normalized orchestrator result:")
        print(json.dumps(response.get("internal_result", {}), indent=2))

        tx = audit_request_transactions(request_date, response)

        if not tx.empty:
            print("Transactions recorded for this request date:")
            print(tx.to_string(index=False))
        else:
            print("Transactions recorded for this request date: none")

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        customer_response = response.get(
            "customer_response",
            "We could not fully process your request."
        )

        print(f"Customer Response: {customer_response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": customer_response,
                "transactions_recorded": 0 if tx.empty else len(tx),
            }
        )

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)

    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    results = run_test_scenarios()
