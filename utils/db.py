"""
utils/db.py
------------
All database setup and data-access functions live here.

We're using SQLite -- a single-file database (no separate server to run).
Python's built-in `sqlite3` module talks to it directly, so no extra
install is needed for the database itself.

Every other page (register.py, admin_events.py, etc.) should import
functions from this file rather than writing SQL directly -- that way
if the schema changes later, there's only one place to update.
"""

import sqlite3
import random
import string
from datetime import datetime
from pathlib import Path

# The .db file will live in the data/ folder, next to this file's project root.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "guest_dashboard.db"


def get_connection():
    """
    Open a connection to the SQLite database file.
    Creates the data/ folder and the .db file automatically if they
    don't exist yet -- nothing to set up by hand.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Foreign keys are OFF by default in SQLite -- this turns them on so
    # a Registration can't point to a Guest or Event that doesn't exist.
    conn.execute("PRAGMA foreign_keys = ON")
    # Lets us access columns by name (row["name"]) instead of just by index.
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_events_columns(conn):
    """
    Migration helper: if the events table already existed from before
    start_time/end_time/tag were added (e.g. you tested an earlier
    version of this app), add the missing columns onto it instead of
    silently ignoring them.

    SQLite's ALTER TABLE can only add one column at a time, and can't
    add a NOT NULL column without a default value on an existing table,
    so these are added as nullable -- old rows just get empty values
    for them, which the UI displays as "--".
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(events)")
    existing_columns = {row["name"] for row in cur.fetchall()}

    if "start_time" not in existing_columns:
        cur.execute("ALTER TABLE events ADD COLUMN start_time TEXT")
    if "end_time" not in existing_columns:
        cur.execute("ALTER TABLE events ADD COLUMN end_time TEXT")
    if "tag" not in existing_columns:
        cur.execute("ALTER TABLE events ADD COLUMN tag TEXT")

    conn.commit()


def init_db():
    """
    Create all four tables if they don't already exist, and make sure
    the events table has every column the app currently expects.
    Safe to call every time the app starts.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS guests (
            guest_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            tag TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS registrations (
            registration_id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL,
            registered_at TEXT NOT NULL,
            ticket_code TEXT NOT NULL UNIQUE,
            checked_in_at TEXT,
            attendance_status TEXT NOT NULL DEFAULT 'registered',
            predicted_no_show REAL,
            FOREIGN KEY (guest_id) REFERENCES guests (guest_id),
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_id INTEGER NOT NULL,
            event_id INTEGER NOT NULL,
            review_text TEXT,
            sentiment_score REAL,
            FOREIGN KEY (guest_id) REFERENCES guests (guest_id),
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        )
    """)

    conn.commit()

    # Handles the case where events already existed before this update.
    _ensure_events_columns(conn)

    conn.close()


# ---------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------

def add_event(event_name, event_date, start_time, end_time, tag):
    """
    Insert a new event.
    event_date: ISO string, e.g. '2026-09-20'
    start_time / end_time: 24-hour string, e.g. '18:30'
    tag: one of the placeholder categories, e.g. 'Movie'
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events (event_name, event_date, start_time, end_time, tag)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event_name, event_date, start_time, end_time, tag),
    )
    conn.commit()
    event_id = cur.lastrowid
    conn.close()
    return event_id


def get_all_events():
    """Return every event, most recent date first -- used to fill dropdowns."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM events ORDER BY event_date DESC")
    rows = cur.fetchall()
    conn.close()
    return rows


def event_has_registrations(event_id):
    """True if at least one guest has registered for this event."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM registrations WHERE event_id = ?", (event_id,))
    count = cur.fetchone()[0]
    conn.close()
    return count > 0


def delete_event(event_id):
    """
    Delete an event, but only if no one has registered for it yet.
    Deleting an event that already has registrations would leave those
    registration rows pointing at nothing, which foreign_keys=ON is
    specifically there to prevent.

    Returns True if the event was deleted, False if it was blocked.
    """
    if event_has_registrations(event_id):
        return False

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------------------
# Guests
# ---------------------------------------------------------------------

def get_guest_by_email(email):
    """
    Look up a guest by exact email match.
    This is a placeholder for now -- real duplicate detection (catching
    'jon@x.com' vs 'jonathan@x.com' or typos) will be added as a separate
    fuzzy-matching step later. For now this only catches an exact repeat.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM guests WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return row


def add_guest(name, email, phone):
    """Insert a new guest and return their new guest_id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO guests (name, email, phone) VALUES (?, ?, ?)",
        (name, email, phone),
    )
    conn.commit()
    guest_id = cur.lastrowid
    conn.close()
    return guest_id


# ---------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------

def _generate_ticket_code(length=5):
    """
    Build a random ticket code like 'LPT43' -- uppercase letters + digits.
    Not guaranteed unique on its own; add_registration() checks that.
    """
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


def _ticket_code_exists(cur, code):
    cur.execute("SELECT 1 FROM registrations WHERE ticket_code = ?", (code,))
    return cur.fetchone() is not None


def add_registration(guest_id, event_id):
    """
    Register a guest for an event: generates a unique ticket code,
    stores the registration, and returns the ticket code to show them.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Keep generating codes until we find one that isn't already taken.
    # With 36^5 (~60 million) possible codes this will basically never
    # loop more than once, but checking is cheap and avoids duplicates.
    code = _generate_ticket_code()
    while _ticket_code_exists(cur, code):
        code = _generate_ticket_code()

    registered_at = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO registrations
            (guest_id, event_id, registered_at, ticket_code, attendance_status)
        VALUES (?, ?, ?, ?, 'registered')
        """,
        (guest_id, event_id, registered_at, code),
    )
    conn.commit()
    conn.close()
    return code