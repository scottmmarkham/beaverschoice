from project_starter import *
from sqlalchemy import create_engine
import pandas as pd

db_engine = create_engine("sqlite:///munder_difflin.db")


def reset_db():
    print("\n=== Resetting database ===")
    init_database(db_engine)


def get_inventory_df():
    return pd.read_sql("SELECT * FROM inventory", db_engine)


def get_transactions_for_date(request_date: str) -> pd.DataFrame:
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


def choose_test_items():
    inventory_df = get_inventory_df()

    in_stock_row = inventory_df.sort_values("current_stock", ascending=False).iloc[0]
    out_of_stock_candidates = [item for item in SUPPORTED_ITEMS if item not in set(inventory_df["item_name"])]
    if not out_of_stock_candidates:
        raise AssertionError("Could not find an out-of-stock item not present in seeded inventory.")

    out_of_stock_item = sorted(out_of_stock_candidates)[0]

    partial_row = inventory_df.sort_values("current_stock", ascending=True).iloc[0]

    return {
        "in_stock_item": in_stock_row["item_name"],
        "in_stock_qty": max(1, int(in_stock_row["current_stock"]) // 2),
        "partial_item": partial_row["item_name"],
        "partial_stock": int(partial_row["current_stock"]),
        "partial_qty": int(partial_row["current_stock"]) + 25,
        "out_of_stock_item": out_of_stock_item,
        "out_of_stock_qty": 40,
        "unsupported_item": "Balloons",
        "unsupported_qty": 12,
    }


def assert_transaction_counts(request_date, expected_sales, expected_stock_orders):
    tx = get_transactions_for_date(request_date)

    sales_count = (tx["transaction_type"] == "sales").sum()
    stock_order_count = (tx["transaction_type"] == "stock_orders").sum()

    assert sales_count == expected_sales, f"Expected {expected_sales} sales transactions, got {sales_count}\n{tx}"
    assert stock_order_count == expected_stock_orders, f"Expected {expected_stock_orders} stock_orders transactions, got {stock_order_count}\n{tx}"


def assert_contains(text: str, expected_substring: str):
    assert expected_substring.lower() in text.lower(), f"Expected '{expected_substring}' in:\n{text}"


def test_unsupported_only():
    reset_db()
    request_date = "2025-01-15"
    request = f"Please send 12 Balloons. (Date of request: {request_date})"

    response = call_your_multi_agent_system(request)
    tx = get_transactions_for_date(request_date)

    assert response["stock_status"] == "unsupported_request"
    assert len(response["recognized_items"]) == 0
    assert len(response["unsupported_items"]) == 1
    assert tx.empty, f"Expected no transactions, got:\n{tx}"

    msg = response.get("customer_message", str(response))
    assert_contains(msg, "unsupported")

    print("PASS: unsupported_only")


def test_fully_in_stock(item_name, qty):
    reset_db()
    request_date = "2025-01-16"
    request = f"I need {qty} {item_name}. (Date of request: {request_date})"

    response = call_your_multi_agent_system(request)
    tx = get_transactions_for_date(request_date)

    assert response["stock_status"] == "ready_to_fulfill"
    assert len(response["recognized_items"]) == 1
    item = response["recognized_items"][0]

    assert item["item_name"] == item_name
    assert item["fulfilled_now"] == qty
    assert item["shortage"] == 0
    assert item["stock_status"] == "in_stock"

    assert_transaction_counts(request_date, expected_sales=1, expected_stock_orders=0)

    sales_row = tx[tx["transaction_type"] == "sales"].iloc[0]
    assert sales_row["item_name"] == item_name
    assert int(sales_row["units"]) == qty

    msg = response.get("customer_message", str(response))
    assert_contains(msg, "fulfilled now")

    print("PASS: fully_in_stock")


def test_partial_stock(item_name, current_stock, requested_qty):
    reset_db()
    request_date = "2025-01-17"
    request = f"Please send {requested_qty} {item_name}. (Date of request: {request_date})"

    response = call_your_multi_agent_system(request)
    tx = get_transactions_for_date(request_date)

    assert response["stock_status"] == "pending_restock"
    assert len(response["recognized_items"]) == 1
    item = response["recognized_items"][0]

    assert item["item_name"] == item_name
    assert item["fulfilled_now"] == current_stock
    assert item["shortage"] == requested_qty - current_stock
    assert item["stock_status"] == "partial_stock"

    assert_transaction_counts(request_date, expected_sales=1, expected_stock_orders=1)

    sales_row = tx[tx["transaction_type"] == "sales"].iloc[0]
    stock_row = tx[tx["transaction_type"] == "stock_orders"].iloc[0]

    assert sales_row["item_name"] == item_name
    assert int(sales_row["units"]) == current_stock

    assert stock_row["item_name"] == item_name
    assert int(stock_row["units"]) == requested_qty - current_stock

    msg = response.get("customer_message", str(response))
    assert_contains(msg, "fulfilled now")
    assert_contains(msg, "pending")
    assert_contains(msg, "restock")

    print("PASS: partial_stock")


def test_out_of_stock(item_name, requested_qty):
    reset_db()
    request_date = "2025-01-18"
    request = f"Can I order {requested_qty} {item_name}? (Date of request: {request_date})"

    response = call_your_multi_agent_system(request)
    tx = get_transactions_for_date(request_date)

    assert response["stock_status"] == "pending_restock"
    assert len(response["recognized_items"]) == 1
    item = response["recognized_items"][0]

    assert item["item_name"] == item_name
    assert item["fulfilled_now"] == 0
    assert item["shortage"] == requested_qty
    assert item["stock_status"] == "out_of_stock"

    assert_transaction_counts(request_date, expected_sales=0, expected_stock_orders=1)

    stock_row = tx[tx["transaction_type"] == "stock_orders"].iloc[0]
    assert stock_row["item_name"] == item_name
    assert int(stock_row["units"]) == requested_qty

    msg = response.get("customer_message", str(response))
    assert_contains(msg, "pending")
    assert_contains(msg, "restock")

    print("PASS: out_of_stock")


def test_parser_edge_cases(in_stock_item):
    parsed = parse_customer_request(
        f"Please, I would like 10 {in_stock_item}; and 5 paper cups! (Date of request: 2025-01-19)"
    )
    assert parsed["requested_date"] == "2025-01-19"
    assert len(parsed["items"]) >= 2, f"Parser missed items: {parsed}"

    parsed2 = parse_customer_request(
        "Can I order 25 card stock and 10 streamers? (Date of request: 2025-01-20)"
    )
    names = [item["item_name"] for item in parsed2["items"]]
    assert "Cardstock" in names, f"Expected Cardstock alias mapping, got {parsed2}"
    assert "Party streamers" in names, f"Expected Party streamers alias mapping, got {parsed2}"

    parsed3 = parse_customer_request(
        "Need 40 A4 paper by Friday and 12 Balloons. (Date of request: 2025-01-21)"
    )
    assert any(item["item_name"] == "A4 paper" for item in parsed3["items"]), f"Expected A4 paper, got {parsed3}"
    assert len(parsed3["unsupported_items"]) == 1, f"Expected unsupported Balloons, got {parsed3}"

    print("PASS: parser_edge_cases")


def run_submission_behavior_tests():
    print("=== Running submission behavior tests ===")
    reset_db()
    chosen = choose_test_items()

    print("Chosen deterministic test items:")
    for k, v in chosen.items():
        print(f"  {k}: {v}")

    test_parser_edge_cases(chosen["in_stock_item"])
    test_unsupported_only()
    test_fully_in_stock(chosen["in_stock_item"], chosen["in_stock_qty"])
    test_partial_stock(chosen["partial_item"], chosen["partial_stock"], chosen["partial_qty"])
    test_out_of_stock(chosen["out_of_stock_item"], chosen["out_of_stock_qty"])

    print("\nALL SUBMISSION-BEHAVIOR TESTS PASSED")


if __name__ == "__main__":
    run_submission_behavior_tests()
