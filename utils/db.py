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
from datetime import datetime, date, timedelta, time as dtime
from difflib import SequenceMatcher
from pathlib import Path

# Used by seed_sample_data() below to give sample reviews real sentiment
# scores instead of made-up placeholder numbers. See utils/ml.py for why
# its own heavy imports (transformers) are deferred rather than done here.
from utils.ml import score_sentiment_batch

# Fuzzy-match threshold for find_possible_duplicate() below, 0-100.
# Higher = stricter (fewer false alarms, but more real duplicates slip
# through). Tune this in one place if it's flagging too much or too
# little once you see it running on real data.
DUPLICATE_MATCH_THRESHOLD = 85

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
    if "predicted_turnout" not in existing_columns:
        cur.execute("ALTER TABLE events ADD COLUMN predicted_turnout REAL")

    conn.commit()


def _ensure_reviews_columns(conn):
    """
    Migration helper: if the reviews table already existed before the
    `rating` column was added (your current .db file does), add it in
    place -- same pattern as the other _ensure_*_columns helpers above.
    Nullable, so existing seeded/real reviews just show "no rating".
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(reviews)")
    existing_columns = {row["name"] for row in cur.fetchall()}

    if "rating" not in existing_columns:
        cur.execute("ALTER TABLE reviews ADD COLUMN rating INTEGER")

    conn.commit()


def _ensure_guests_columns(conn):
    """
    Migration helper: if the guests table already existed from before
    possible_duplicate_of/duplicate_match_score were added (your
    current .db file does), add the missing columns onto it instead of
    silently ignoring them. Same idea as _ensure_events_columns above --
    added as nullable, so existing guest rows just get an empty value
    (meaning "not flagged"), which is exactly what we want for them.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(guests)")
    existing_columns = {row["name"] for row in cur.fetchall()}

    if "possible_duplicate_of" not in existing_columns:
        cur.execute("ALTER TABLE guests ADD COLUMN possible_duplicate_of INTEGER")
    if "duplicate_match_score" not in existing_columns:
        cur.execute("ALTER TABLE guests ADD COLUMN duplicate_match_score REAL")
    if "segment" not in existing_columns:
        cur.execute("ALTER TABLE guests ADD COLUMN segment TEXT")

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
            phone TEXT,
            possible_duplicate_of INTEGER,
            duplicate_match_score REAL,
            segment TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            tag TEXT,
            predicted_turnout REAL
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
            rating INTEGER,
            FOREIGN KEY (guest_id) REFERENCES guests (guest_id),
            FOREIGN KEY (event_id) REFERENCES events (event_id)
        )
    """)

    # Simple key/value store for small admin toggles that need to persist
    # across browser sessions -- e.g. the "bypass review eligibility"
    # testing switch. st.session_state wouldn't work for this, since it
    # only lives inside one browser tab, and a guest testing the review
    # link would be in a different tab/session than the admin who set it.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.commit()

    # Handles the case where events/guests/reviews already existed before this update.
    _ensure_events_columns(conn)
    _ensure_guests_columns(conn)
    _ensure_reviews_columns(conn)

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
    This only catches an exact repeat (e.g. the same guest registering
    twice with the same email). Catching a *different* email or a typo
    in the name is what find_possible_duplicate(), below, is for.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM guests WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    return row


def add_guest(name, email, phone, possible_duplicate_of=None, duplicate_match_score=None):
    """
    Insert a new guest and return their new guest_id.

    possible_duplicate_of / duplicate_match_score are optional. Pass
    them when find_possible_duplicate() found a likely match, so this
    new guest row is flagged for the admin to review later. Leave both
    as None (the default) for a normal, unflagged guest.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO guests (name, email, phone, possible_duplicate_of, duplicate_match_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, email, phone, possible_duplicate_of, duplicate_match_score),
    )
    conn.commit()
    guest_id = cur.lastrowid
    conn.close()
    return guest_id


def _normalize_name_for_matching(name):
    """
    Lowercase a name and sort its words alphabetically.
    This is what lets "Rugved Patil" and "Patil Rugved" still be
    recognized as the same name even though the words are swapped --
    comparing the raw strings directly would score that pair as very
    different, since string comparison cares about order.
    """
    words = name.strip().lower().split()
    return " ".join(sorted(words))


def _name_similarity(name_a, name_b):
    """
    Return a 0-100 score for how similar two names are (100 = identical
    after normalizing, 0 = nothing alike).

    Built on difflib, which ships with Python -- no extra install
    needed. SequenceMatcher.ratio() works by finding the longest
    stretches of matching characters between two strings, so small
    typos ("Rugved" vs "Rugved" with one letter swapped) still score
    high, while genuinely different names score low.
    """
    a = _normalize_name_for_matching(name_a)
    b = _normalize_name_for_matching(name_b)
    return SequenceMatcher(None, a, b).ratio() * 100


def find_possible_duplicate(name, exclude_guest_id=None, threshold=DUPLICATE_MATCH_THRESHOLD):
    """
    Fuzzy-check a name against every existing guest's name.

    Returns a dict like {"guest_id": ..., "name": ..., "score": ...}
    for the closest match, if it scores at or above `threshold`.
    Returns None if nothing scores high enough.

    exclude_guest_id skips one guest_id while checking -- not used yet,
    but there so this function can be reused later (e.g. on an "edit
    guest" screen) without a guest getting flagged as a duplicate of
    themselves.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT guest_id, name FROM guests")
    all_guests = cur.fetchall()
    conn.close()

    best_match = None
    best_score = 0

    for guest in all_guests:
        if exclude_guest_id is not None and guest["guest_id"] == exclude_guest_id:
            continue
        score = _name_similarity(name, guest["name"])
        if score > best_score:
            best_score = score
            best_match = guest

    if best_match is not None and best_score >= threshold:
        return {
            "guest_id": best_match["guest_id"],
            "name": best_match["name"],
            "score": round(best_score, 1),
        }
    return None


def get_or_create_guest(name, email, phone):
    """
    The full "find this guest, or create them" logic in one place:
    exact email match wins first; failing that, a fuzzy name match
    flags the new guest as a possible duplicate; failing that, it's
    just a normal new guest.

    This is exactly what register.py needs, and it's also what the
    sample-data seeder needs (see seed_sample_data() below) -- pulling
    it out here means both call one function instead of the same
    three-way logic being copy-pasted in two places.

    Returns the resulting guest_id either way.
    """
    existing = get_guest_by_email(email)
    if existing:
        return existing["guest_id"]

    possible_match = find_possible_duplicate(name)
    if possible_match:
        return add_guest(
            name, email, phone,
            possible_duplicate_of=possible_match["guest_id"],
            duplicate_match_score=possible_match["score"],
        )
    return add_guest(name, email, phone)


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


def add_registration(guest_id, event_id, attendance_status="registered",
                      checked_in_at=None, registered_at=None):
    """
    Register a guest for an event: generates a unique ticket code,
    stores the registration, and returns the ticket code to show them.

    attendance_status / checked_in_at / registered_at are optional and
    only matter for the sample-data seeder below -- a real guest
    registering through the form always gets the defaults (registered
    now, not checked in yet), so register.py doesn't need to change.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Keep generating codes until we find one that isn't already taken.
    # With 36^5 (~60 million) possible codes this will basically never
    # loop more than once, but checking is cheap and avoids duplicates.
    code = _generate_ticket_code()
    while _ticket_code_exists(cur, code):
        code = _generate_ticket_code()

    if registered_at is None:
        registered_at = datetime.now().isoformat(timespec="seconds")

    cur.execute(
        """
        INSERT INTO registrations
            (guest_id, event_id, registered_at, ticket_code, attendance_status, checked_in_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (guest_id, event_id, registered_at, code, attendance_status, checked_in_at),
    )
    conn.commit()
    conn.close()
    return code


def get_registration_by_ticket(ticket_code):
    """
    Look up a single registration by its ticket code, joined with the
    guest's name and the event's name/date -- everything review.py
    needs to identify who's reviewing what, in one query instead of
    three separate lookups.

    Returns a row with columns: registration_id, guest_id, event_id,
    ticket_code, attendance_status, checked_in_at, guest_name,
    event_name, event_date. Returns None if no registration has that
    ticket code.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            r.registration_id, r.guest_id, r.event_id, r.ticket_code,
            r.attendance_status, r.checked_in_at,
            g.name AS guest_name,
            e.event_name, e.event_date
        FROM registrations r
        JOIN guests g ON g.guest_id = r.guest_id
        JOIN events e ON e.event_id = r.event_id
        WHERE r.ticket_code = ?
        """,
        (ticket_code,),
    )
    row = cur.fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------
# Attendance / check-in
# ---------------------------------------------------------------------

def mark_attendance_by_ticket(ticket_code):
    """
    Look up a registration by ticket code and, if it isn't already
    marked "attended", mark it attended and stamp checked_in_at with
    the current date/time.

    Returns a dict describing what happened, so admin_attendance.py
    can show the right message without doing its own lookups:
      {"status": "not_found"}
      {"status": "already_checked_in", "registration": <row>}
      {"status": "checked_in", "registration": <row>}
    "registration" (when present) is a fresh get_registration_by_ticket()
    row, so the page can show the guest's name, event, and timestamp.

    Note: this doesn't check the event date at all -- unlike reviews,
    check-in has no "too early" rule here, since you'd normally only be
    running this page live at the event itself.
    """
    registration = get_registration_by_ticket(ticket_code)
    if registration is None:
        return {"status": "not_found"}

    if registration["attendance_status"] == "attended":
        return {"status": "already_checked_in", "registration": registration}

    conn = get_connection()
    cur = conn.cursor()
    checked_in_at = datetime.now().isoformat(timespec="seconds")
    cur.execute(
        """
        UPDATE registrations
        SET attendance_status = 'attended', checked_in_at = ?
        WHERE registration_id = ?
        """,
        (checked_in_at, registration["registration_id"]),
    )
    conn.commit()
    conn.close()

    updated_registration = get_registration_by_ticket(ticket_code)
    return {"status": "checked_in", "registration": updated_registration}


def get_all_registrations_detailed():
    """
    Return every registration, joined with the guest's details and the
    event's details -- one row per registration (a guest with 3
    registrations appears 3 times), most recently registered first.

    This is what admin_manage.py builds its table from: it has the
    ticket code and attendance status (which live on the registration)
    right alongside the guest's name/email/phone and the event's
    name/date/tag, so nothing needs a second lookup. Also includes
    predicted_no_show, which admin_analytics.py uses to display the
    no-show model's saved predictions alongside guest/event names.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            r.registration_id, r.ticket_code, r.attendance_status,
            r.registered_at, r.checked_in_at, r.predicted_no_show,
            g.guest_id, g.name AS guest_name, g.email, g.phone,
            g.possible_duplicate_of, g.duplicate_match_score, g.segment,
            e.event_id, e.event_name, e.event_date, e.tag
        FROM registrations r
        JOIN guests g ON g.guest_id = r.guest_id
        JOIN events e ON e.event_id = r.event_id
        ORDER BY r.registered_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_registrations_for_noshow_model():
    """
    Return every registration with just what the no-show model (see
    utils/ml.py) needs: which guest, which event (+ its tag and date),
    when they registered, and the outcome (attendance_status) if it's
    already known.

    Unlike get_all_registrations_detailed() above, this includes
    predicted_no_show and leaves out guest/event display fields
    (email, phone, event name...) the model has no use for.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            r.registration_id, r.guest_id, r.event_id,
            r.attendance_status, r.registered_at, r.predicted_no_show,
            e.event_date, e.tag
        FROM registrations r
        JOIN events e ON e.event_id = r.event_id
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def bulk_set_predicted_no_show(predictions):
    """
    Save predicted no-show probabilities for a batch of registrations
    at once. `predictions` is a list of (registration_id, probability)
    tuples -- probability is a float from 0 (certain to attend) to 1
    (certain to no-show).

    Uses executemany() (one round-trip for the whole batch) rather
    than looping and calling this once per row, same reasoning as
    score_sentiment_batch() being faster than scoring reviews one by
    one.
    """
    if not predictions:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE registrations SET predicted_no_show = ? WHERE registration_id = ?",
        [(prob, reg_id) for (reg_id, prob) in predictions],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Guest segmentation
# ---------------------------------------------------------------------

def get_guests_for_segmentation():
    """
    Return every guest with the two counts utils/ml.py's segment_guests()
    needs: how many of their registrations are "resolved" (the event
    already happened, so the real outcome -- attended or no_show -- is
    known), and how many of those were attended. Also includes the
    guest's name/email and their current saved segment, so this same
    query can feed both the model (which only cares about the counts)
    and admin_analytics.py's results table (which needs the display
    fields too) without querying twice.

    LEFT JOIN (not JOIN) so a guest with zero registrations still shows
    up as one row with resolved_count = 0, rather than being dropped
    entirely -- that's exactly the case segment_guests() tags "New".
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            g.guest_id, g.name AS guest_name, g.email, g.segment,
            COUNT(CASE WHEN r.attendance_status IN ('attended', 'no_show') THEN 1 END)
                AS resolved_count,
            COUNT(CASE WHEN r.attendance_status = 'attended' THEN 1 END)
                AS attended_count
        FROM guests g
        LEFT JOIN registrations r ON r.guest_id = g.guest_id
        GROUP BY g.guest_id, g.name, g.email, g.segment
        ORDER BY g.name
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def bulk_set_segment(assignments):
    """
    Save segment labels ("VIP" / "Regular" / "New") for a batch of
    guests at once. `assignments` is a list of (guest_id, segment)
    tuples -- same executemany-in-one-batch approach as
    bulk_set_predicted_no_show() above, for the same reason (one
    round-trip instead of one UPDATE per guest).
    """
    if not assignments:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE guests SET segment = ? WHERE guest_id = ?",
        [(segment, guest_id) for (guest_id, segment) in assignments],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Turnout forecasting
# ---------------------------------------------------------------------

def get_events_for_forecast():
    """
    Return every event with what utils/ml.py's train_and_predict_turnout()
    needs (registered_count, attended_count, resolved_count, event_date)
    plus display fields (event_name, tag, predicted_turnout) for
    admin_analytics.py's chart/table -- same "one query, two
    consumers" approach as get_guests_for_segmentation() above.

    resolved_count == registered_count (and > 0) is how an event is
    recognized as "fully resolved" -- every one of its registrations
    has a known outcome (attended/no_show), so its true turnout rate
    can be computed. If resolved_count < registered_count, at least
    one registration is still sitting at "registered" (check-in never
    happened, even though other guests at the same event were
    resolved) -- too ambiguous to treat as ground truth, so it's left
    out of training rather than guessed at.

    LEFT JOIN so an event with zero registrations still appears (as
    registered_count = 0) instead of being silently dropped.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            e.event_id, e.event_name, e.event_date, e.tag, e.predicted_turnout,
            COUNT(r.registration_id) AS registered_count,
            COUNT(CASE WHEN r.attendance_status = 'attended' THEN 1 END)
                AS attended_count,
            COUNT(CASE WHEN r.attendance_status IN ('attended', 'no_show') THEN 1 END)
                AS resolved_count
        FROM events e
        LEFT JOIN registrations r ON r.event_id = e.event_id
        GROUP BY e.event_id, e.event_name, e.event_date, e.tag, e.predicted_turnout
        ORDER BY e.event_date
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def bulk_set_predicted_turnout(predictions):
    """
    Save predicted turnout rates for a batch of events at once.
    `predictions` is a list of (event_id, rate) tuples, rate being a
    float from 0 (nobody expected to show) to 1 (everyone expected to
    show) -- same batch-executemany approach as bulk_set_predicted_no_show()
    and bulk_set_segment() above.
    """
    if not predictions:
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "UPDATE events SET predicted_turnout = ? WHERE event_id = ?",
        [(rate, event_id) for (event_id, rate) in predictions],
    )
    conn.commit()
    conn.close()


def has_review(guest_id, event_id):
    """True if this guest has already left a review for this event."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM reviews WHERE guest_id = ? AND event_id = ?",
        (guest_id, event_id),
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def is_review_eligible(registration, bypass=False):
    """
    Decide whether the guest on this registration is allowed to leave
    a review right now.

    Real rule: they were marked "attended" AND the event's date has
    already passed (not today, not in the future) -- matches the
    original scope doc ("only guests marked as attended").

    bypass=True skips both checks entirely. Pass the current value of
    the "Bypass review eligibility (testing)" setting from the Data
    Tools page here -- it exists because attendance/check-in isn't
    fully wired up yet (admin_attendance.py is still a stub), so there'd
    otherwise be no way to test this page end-to-end yet.
    """
    if bypass:
        return True
    if registration["attendance_status"] != "attended":
        return False
    event_date = date.fromisoformat(registration["event_date"])
    return event_date < date.today()


def add_review(guest_id, event_id, review_text, sentiment_score=None, rating=None):
    """
    Insert a review left by a guest for an event they attended.

    sentiment_score is optional -- leave it as None for a real guest
    review, since scoring it is a separate, not-yet-built step that
    runs the text through VADER or similar. The sample-data seeder
    below passes an approximate score directly so the analytics page
    has something to chart before that's built.

    rating is the 1-5 star rating from the review form, also optional
    so old code that calls add_review() without one still works.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reviews (guest_id, event_id, review_text, sentiment_score, rating)
        VALUES (?, ?, ?, ?, ?)
        """,
        (guest_id, event_id, review_text, sentiment_score, rating),
    )
    conn.commit()
    review_id = cur.lastrowid
    conn.close()
    return review_id


def get_all_reviews_detailed():
    """
    Return every review, joined with the guest's name and the event's
    name/date/tag -- one row per review, most recent first.

    This is what admin_analytics.py builds its sentiment charts and
    table from, same idea as get_all_registrations_detailed() above.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            rv.review_id, rv.review_text, rv.sentiment_score, rv.rating,
            g.guest_id, g.name AS guest_name,
            e.event_id, e.event_name, e.event_date, e.tag
        FROM reviews rv
        JOIN guests g ON g.guest_id = rv.guest_id
        JOIN events e ON e.event_id = rv.event_id
        ORDER BY rv.review_id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------
# Settings (small admin toggles, stored in the DB so they persist
# across browser sessions -- unlike st.session_state)
# ---------------------------------------------------------------------

def get_setting(key, default=None):
    """Return the stored value for `key`, or `default` if it's not set."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key, value):
    """
    Store `value` under `key`, overwriting whatever was there before.
    "INSERT ... ON CONFLICT DO UPDATE" (an "upsert") means we don't have
    to check whether the key already exists first -- SQLite handles the
    insert-or-update in one step.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def get_bypass_setting():
    """True if the Data Tools 'bypass review eligibility' toggle is on."""
    return get_setting("bypass_review_eligibility", "0") == "1"


def set_bypass_setting(enabled):
    """Turn the review-eligibility testing bypass on or off."""
    set_setting("bypass_review_eligibility", "1" if enabled else "0")


# ---------------------------------------------------------------------
# Admin data tools (reset + sample data)
# ---------------------------------------------------------------------

def get_table_counts():
    """
    Return the number of rows in each table, e.g.
    {"guests": 30, "events": 15, "registrations": 210, "reviews": 74}.
    Used by the Data Tools page so you can see what's there before
    resetting it, and confirm what got added after seeding.
    """
    conn = get_connection()
    cur = conn.cursor()
    counts = {}
    for table in ("guests", "events", "registrations", "reviews"):
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cur.fetchone()[0]
    conn.close()
    return counts


def reset_all_data():
    """
    The "kill switch": delete every row from all four tables and reset
    SQLite's auto-increment counters, so the next guest/event created
    after this starts back at ID 1 instead of continuing from wherever
    the old data left off. Irreversible -- the page calling this is
    responsible for getting the admin to confirm first.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Order doesn't strictly matter here since we're clearing every
    # table, but deleting registrations/reviews (which point at guests
    # and events) before guests/events keeps foreign_keys=ON happy at
    # every step instead of only at the end.
    cur.execute("DELETE FROM reviews")
    cur.execute("DELETE FROM registrations")
    cur.execute("DELETE FROM guests")
    cur.execute("DELETE FROM events")
    # sqlite_sequence is SQLite's internal "next AUTOINCREMENT value"
    # tracker, one row per table. Clearing it means new rows start
    # back at 1 instead of picking up from the highest ID ever used.
    cur.execute("DELETE FROM sqlite_sequence")
    conn.commit()
    conn.close()


# --- sample data building blocks -------------------------------------

_EVENT_NAMES_BY_TAG = {
    "Movie": ["Movie Premiere Night", "Classic Film Screening",
              "Indie Cinema Showcase", "Late Night Movie Marathon"],
    "Play": ["Broadway Night", "Community Theatre Play",
             "Shakespeare in the Park", "One-Act Play Festival"],
    "Sports": ["City Marathon", "Local Cricket Match",
               "Basketball Tournament", "Charity Football Cup"],
    "Dance": ["Salsa Night", "Contemporary Dance Recital",
              "Hip-Hop Showcase", "Ballroom Dance Gala"],
}

# How much each event tag nudges attendance up or down, on top of a
# guest's own reliability (see guest_reliability in seed_sample_data
# below). E.g. free/outdoor Sports events realistically get skipped
# more often than a ticketed Play. This exists purely so the no-show
# model (utils/ml.py) has a second real, non-guest-specific pattern to
# find in the sample data besides "this particular guest tends to flake."
_TAG_ATTEND_SHIFT = {
    "Movie": 0.05,
    "Play": 0.03,
    "Sports": -0.12,
    "Dance": 0.0,
}


_FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai",
                "Reyansh", "Ayaan", "Ishaan", "Kabir", "Ananya", "Diya",
                "Saanvi", "Aadhya", "Kiara", "Myra", "Pari", "Anika",
                "Navya", "Riya"]
_LAST_NAMES = ["Sharma", "Verma", "Gupta", "Patil", "Reddy", "Nair",
               "Iyer", "Singh", "Rao", "Deshmukh", "Kulkarni", "Joshi",
               "Mehta", "Kapoor"]

_REVIEWS_POSITIVE = [
    "Amazing event, loved every moment of it!",
    "Really well organized and so much fun.",
    "One of the best events I've been to this year.",
    "Great vibe, great people, would definitely come again.",
    "Everything ran smoothly, had a wonderful time.",
]
_REVIEWS_NEUTRAL = [
    "It was okay, nothing special but not bad either.",
    "Decent event, could have been better organized.",
    "Average experience overall.",
]
_REVIEWS_NEGATIVE = [
    "Quite disappointing, expected a lot more.",
    "Poorly organized, long wait times and confusion.",
    "Not worth it, wouldn't recommend.",
]

# Star ratings a real guest would plausibly leave alongside each kind
# of review text -- e.g. someone writing a clearly positive review
# almost always leaves 4 or 5 stars, not 2. random.choice() picking
# from these short weighted lists gives seeded reviews a rating that
# actually matches their tone, instead of leaving rating empty like
# before.
_RATINGS_BY_BUCKET = {
    "positive": [5, 5, 5, 4],
    "neutral": [3, 3, 2, 4],
    "negative": [1, 1, 2],
}


def _pick_review():
    """Pick one (text, rating) pair for a seeded review, weighted
    toward positive -- most real post-event reviews skew positive,
    since happy attendees are more likely to bother leaving one at all.

    Note: this used to also hand back a made-up sentiment_score
    alongside the text. That's gone now -- seed_sample_data() below
    runs every generated review through the real sentiment model
    instead, the same one real guest reviews go through, so the
    Analytics page shows genuine model output either way rather than a
    mix of real and hand-picked numbers. The star rating returned here
    is still a plausible guess (there's no model for that), matched to
    whichever bucket the text came from."""
    bucket_name, bucket_texts = random.choices(
        [
            ("positive", _REVIEWS_POSITIVE),
            ("neutral", _REVIEWS_NEUTRAL),
            ("negative", _REVIEWS_NEGATIVE),
        ],
        weights=[0.55, 0.25, 0.20],
        k=1,
    )[0]
    text = random.choice(bucket_texts)
    rating = random.choice(_RATINGS_BY_BUCKET[bucket_name])
    return text, rating


def _make_similar_name(name):
    """
    Build a "near-duplicate" of a name -- either the words reordered
    or one letter changed -- so the sample data includes a few guests
    that find_possible_duplicate() should actually catch. Makes the
    Data Tools output more useful as a demo of that feature too.
    """
    if random.random() < 0.5 and " " in name:
        words = name.split()
        random.shuffle(words)
        return " ".join(words)
    chars = list(name)
    idx = random.randrange(len(chars))
    if chars[idx].isalpha():
        chars[idx] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def seed_sample_data():
    """
    Populate the database with a realistic-looking demo dataset:
    10-20 events spread across the past and future, a pool of guests
    registered across them, a believable mix of attended/no-show
    outcomes for past events, and reviews (with sentiment scores) from
    some of the guests who attended.

    Safe to call more than once -- it adds on top of whatever's
    already there rather than replacing it. Use reset_all_data() first
    if you want a clean slate before seeding.

    Returns a dict of how many rows of each kind were added, e.g.
    {"events": 15, "guests": 32, "registrations": 187, "reviews": 71}.
    """
    added = {"events": 0, "guests": 0, "registrations": 0, "reviews": 0}

    # --- Events: spread from 6 months ago to 2 months from now, so
    # there's a healthy mix of past events (needed later for no-show
    # prediction and forecasting, which train on history) and future
    # ones (needed to demo "upcoming event" views). ---
    today = date.today()
    window_start = today - timedelta(days=180)
    window_end = today + timedelta(days=60)
    window_days = (window_end - window_start).days

    num_events = random.randint(10, 20)
    events = []  # list of (event_id, event_date, is_past)

    for _ in range(num_events):
        tag = random.choice(list(_EVENT_NAMES_BY_TAG.keys()))
        name = random.choice(_EVENT_NAMES_BY_TAG[tag])
        event_date = window_start + timedelta(days=random.randint(0, window_days))
        start_hour = random.randint(10, 20)
        start_minute = random.choice([0, 15, 30, 45])
        start_time = f"{start_hour:02d}:{start_minute:02d}"
        end_hour = min(start_hour + random.randint(1, 3), 23)
        end_time = f"{end_hour:02d}:{start_minute:02d}"

        event_id = add_event(name, event_date.isoformat(), start_time, end_time, tag)
        events.append((event_id, event_date, event_date < today, tag))
        added["events"] += 1

    # --- Guests: a pool of fresh names, plus a handful of deliberate
    # near-duplicates so the fuzzy-match flagging has something to
    # catch in the sample data too. ---
    num_guests = random.randint(25, 35)
    guest_ids = []
    generated_names = []

    for i in range(num_guests):
        # Every ~8th guest is a near-duplicate of an earlier one instead
        # of a fresh name, once there's at least one name to riff on.
        if generated_names and i % 8 == 7:
            name = _make_similar_name(random.choice(generated_names))
        else:
            name = f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
            generated_names.append(name)

        email = f"{name.lower().replace(' ', '.')}{i}@example.com"
        phone = f"9{random.randint(100000000, 999999999)}"

        guest_id = get_or_create_guest(name, email, phone)
        guest_ids.append(guest_id)
        added["guests"] += 1

    # Each guest gets a hidden "reliability" -- their personal tendency
    # to actually show up -- used below to decide attended/no_show for
    # past events. This is what makes no-show prediction meaningful:
    # without it, every guest would have the exact same flat chance of
    # attending regardless of their event history, and a model trained
    # on that data couldn't learn anything real. random.betavariate(6, 2)
    # picks values between 0 and 1 shaped so most land near 0.75 (our
    # overall target attendance rate) while some guests are naturally
    # more or less reliable than others. This number itself is never
    # stored anywhere -- the model has to infer it purely from each
    # guest's visible past attendance, same as it would with real guests.
    guest_reliability = {
        guest_id: random.betavariate(6, 2) for guest_id in guest_ids
    }

    # --- Registrations (+ reviews for attended guests on past events) ---
    # Reviews aren't inserted immediately -- their text/rating get
    # collected into pending_reviews first, and all the text gets
    # scored together in one batch call below (after this loop),
    # instead of loading the sentiment model and scoring one at a time
    # as we go. See score_sentiment_batch() in utils/ml.py for why
    # that's faster.
    pending_reviews = []  # list of (guest_id, event_id, review_text, rating)

    for event_id, event_date, is_past, tag in events:
        attendee_count = random.randint(max(1, num_guests // 3), int(num_guests * 0.7))
        attendees = random.sample(guest_ids, k=min(attendee_count, len(guest_ids)))

        for guest_id in attendees:
            # event_date/registered_date are plain `date` objects (no time
            # component), so we combine each with a random time-of-day to
            # get a full datetime before turning it into a stored string.
            registered_date = event_date - timedelta(days=random.randint(1, 30))
            registered_at = datetime.combine(
                registered_date, dtime(random.randint(9, 21), random.choice([0, 15, 30, 45]))
            ).isoformat(timespec="seconds")

            if is_past:
                # This guest's own reliability, nudged up or down by
                # how flaky this event's tag tends to be -- clamped so
                # every guest still has *some* chance of going either
                # way, however reliable or unreliable they are.
                attend_probability = guest_reliability[guest_id] + _TAG_ATTEND_SHIFT.get(tag, 0.0)
                attend_probability = min(0.97, max(0.03, attend_probability))

                if random.random() < attend_probability:
                    status = "attended"
                    checked_in_at = datetime.combine(
                        event_date, dtime(random.randint(9, 21), random.choice([0, 15, 30, 45]))
                    ).isoformat(timespec="seconds")
                else:
                    status = "no_show"
                    checked_in_at = None
            else:
                # Event hasn't happened yet -- can't be attended/no-show.
                status = "registered"
                checked_in_at = None

            add_registration(
                guest_id, event_id,
                attendance_status=status,
                checked_in_at=checked_in_at,
                registered_at=registered_at,
            )
            added["registrations"] += 1

            # Only attended guests can leave a review, and even then
            # not everyone bothers -- about half do.
            if status == "attended" and random.random() < 0.5:
                text, rating = _pick_review()
                pending_reviews.append((guest_id, event_id, text, rating))

    # Score every pending review's text in one batch call, then insert
    # them all with their real sentiment_score (+ the plausible rating
    # picked alongside the text above). If nothing qualified for a
    # review (e.g. no past events yet), skip this entirely so we don't
    # load the sentiment model for no reason.
    if pending_reviews:
        texts = [text for (_, _, text, _) in pending_reviews]
        scores = score_sentiment_batch(texts)
        for (guest_id, event_id, text, rating), score in zip(pending_reviews, scores):
            add_review(guest_id, event_id, text, sentiment_score=score, rating=rating)
            added["reviews"] += 1

    return added