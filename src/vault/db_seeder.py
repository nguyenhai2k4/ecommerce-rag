import sqlite3
from pathlib import Path
from typing import Optional
from src.config import config


def get_database_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    target_path = db_path or config.db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def seed_database(db_path: Optional[Path] = None, force: bool = False) -> Path:
    """
    Initializes and seeds the SQLite retail database.
    If the database already exists and has records, seeding is skipped unless force=True.
    """
    target_path = db_path or config.db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path))
    cursor = conn.cursor()

    # Check if tables already exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
    table_exists = cursor.fetchone() is not None

    if table_exists and not force:
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        if count > 0:
            conn.close()
            return target_path

    # Read and apply schema
    schema_file = config.structured_dir / "schema.sql"
    if schema_file.exists():
        with open(schema_file, "r", encoding="utf-8") as f:
            cursor.executescript(f.read())

    # Seed Products
    products = [
        (1, "SKU-AUD-001", "AeroSound Pro Wireless ANC Headphones", "Audio", 199.99, 12, 12, "Active"),
        (2, "SKU-AUD-002", "AeroBuds True Wireless In-Ear Earphones", "Audio", 89.99, 45, 12, "Active"),
        (3, "SKU-KEY-002", "Ergonomic Mechanical Keyboard Pro", "Peripherals", 149.50, 30, 24, "Active"),
        (4, "SKU-SMT-004", "PulseFit Smart Tracker Band", "Wearables", 79.99, 68, 12, "Active"),
        (5, "SKU-PWR-007", "UltraCharge 65W GaN Fast Charger", "Power", 39.99, 8, 18, "Active"),
        (6, "SKU-ACC-010", "Universal USB-C Magnetic Braided Cable (2m)", "Accessories", 19.99, 120, 12, "Active"),
    ]
    cursor.executemany(
        "INSERT INTO products (product_id, sku, name, category, price, stock_quantity, warranty_months, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        products,
    )

    # Seed Customers
    customers = [
        (101, "Alice Chen", "alice.chen@techcorp.io", "VIP", "San Francisco", "2024-02-15"),
        (102, "Bob Martinez", "bob.martinez@gmail.com", "Standard", "Austin", "2025-06-10"),
        (103, "Catherine Dupont", "c.dupont@financehub.com", "Gold", "New York", "2024-11-01"),
        (104, "David Kim", "david.kim@startup.co", "Standard", "Seattle", "2026-01-20"),
        (105, "Elena Rostova", "elena.rostova@designworks.org", "VIP", "Chicago", "2023-09-12"),
    ]
    cursor.executemany(
        "INSERT INTO customers (customer_id, name, email, tier, city, registered_date) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        customers,
    )

    # Seed Orders
    # Delivery dates set relative to September/October 2026 for realistic multi-hop return window checks:
    # Alice (VIP): Delivered Sept 10, 2026 (42 days ago relative to Oct 22, 2026 -> eligible for 60-day VIP return)
    # Bob (Standard): Delivered Sept 14, 2026 (38 days ago -> ineligible for 30-day Standard return)
    orders = [
        (5001, 101, 1, "2026-09-06", "2026-09-10", 1, 199.99, 199.99, "Delivered"),
        (5002, 102, 1, "2026-09-10", "2026-09-14", 1, 199.99, 199.99, "Delivered"),
        (5003, 103, 2, "2026-10-10", "2026-10-14", 1, 89.99, 89.99, "Delivered"),
        (5004, 104, 5, "2026-10-18", "2026-10-21", 2, 39.99, 79.98, "Delivered"),
        (5005, 105, 3, "2026-10-20", None, 1, 149.50, 149.50, "In Transit"),
        (5006, 101, 5, "2026-10-15", "2026-10-17", 1, 39.99, 39.99, "Delivered"),
    ]
    cursor.executemany(
        "INSERT INTO orders (order_id, customer_id, product_id, order_date, delivery_date, quantity, unit_price, total_amount, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        orders,
    )

    # Seed Inventory Logs
    logs = [
        (1, 1, "Restock", 50, "Initial Q3 shipment from Shenzhen factory", "2026-07-01 09:00:00"),
        (2, 5, "Restock", 100, "Bulk shipment arrival", "2026-07-15 14:30:00"),
        (3, 1, "Customer Return", 1, "Customer returned sealed unit - restocked", "2026-08-10 11:15:00"),
        (4, 5, "Quality Hold", 5, "Inspection for high thermal load reports", "2026-09-05 16:00:00"),
    ]
    cursor.executemany(
        "INSERT INTO inventory_logs (log_id, product_id, event_type, units, notes, logged_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        logs,
    )

    conn.commit()
    conn.close()
    return target_path


if __name__ == "__main__":
    db = seed_database(force=True)
    print(f"Database successfully seeded at: {db}")
