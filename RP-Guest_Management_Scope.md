# RP-Guest-Management — Project Scope

*Current as of the v2 (login + roles) build. Supersedes any earlier scope notes written before login existed.*

## 1. Overview

A Streamlit dashboard for managing guests across recurring, ticketed events, from creation through registration, check-in, and post-event feedback -- with a role-based login sitting in front of all of it. Beyond core CRUD functionality, the project layers in AI/Data Science features (clustering, prediction, NLP, forecasting) to serve as a portfolio piece demonstrating applied ML skills in a practical, real-world context.

## 2. Roles & Access

Every page is reachable only by the role(s) it's built for, both by what `app.py` hands to `st.navigation()` and by a `require_role()` check at the top of the page itself.

| Role | Can do | Scope |
|---|---|---|
| **Admin** | Create events, view/search/edit every guest and registration, resolve flagged duplicates, check guests in, view analytics, seed/reset sample data, toggle the review-eligibility testing bypass | System-wide -- every event, every guest |
| **Organizer** | Create/edit/delete events, check guests in, view a read-only guest list, view analytics | Scoped to events they personally created (`events.created_by`) |
| **Guest** | Browse and register for events, cancel their own registrations, edit their profile, leave reviews | Scoped to their own account (`guests.user_id`) |

33 accounts are pre-loaded automatically on first run: 1 Admin, 7 Organizers, 25 Guests, all sharing one demo password. There's no signup flow -- see Out of Scope.

## 3. End-to-End Workflow

1. **Event Creation** — An Admin or Organizer creates an event (name, date, time, tag) before any guest can book it.
2. **Guest Registration ("Booking")** — A logged-in guest browses open events on **My Events** and registers with one click (no form fields -- their profile is already known). A unique alphanumeric ticket code (e.g. `LPT43`) is generated and shown on screen -- this is the guest's ticket, and doubles as their check-in code.
3. **Storage** — The registration, including the ticket code, is stored and linked to the guest and event.
4. **Duplicate Check** — New guest profiles are checked (fuzzy match, via `difflib`) against existing records to avoid duplicate profiles; Admin reviews and resolves any flagged pairs.
5. **No-Show Prediction** — Before the event, a model flags guests likely not to show up, using registration + guest history data.
6. **Admin/Organizer Management** — Admin can view, search, and edit any guest/registration; Organizer sees the same for their own events (read-only guest list).
7. **Check-In** — At the door, the ticket code is typed into a check-in screen (Admin or Organizer, matching who owns the event); the system validates it and marks that registration attended, with a timestamp.
8. **Post-Event Review** — Once a registration is marked attended and the event date has passed, that event shows up in the guest's **Leave a Review** dropdown. The guest picks it, rates it, and writes feedback -- no code to type, since they're already logged in. A guest's own reviews never show up twice; an already-reviewed event drops off the list immediately.
9. **Sentiment Analysis** — Review text is run through a HuggingFace sentiment classifier.
10. **Summary & Analytics** — Sentiment trends, attendance rates, and guest segments are summarized on the analytics dashboard -- system-wide for Admin, scoped to owned events for Organizer.

## 4. Core Features

### 4.1 Login & Roles
Pre-loaded accounts only, no self-registration. Session state (`utils/auth.py`) tracks who's logged in; `require_role()` gates every page as a backstop to the navigation menu itself only listing pages a role should see.

### 4.2 Event Creation
Admin or Organizer creates an event -- name, date, start/end time, and a tag (`Movie` / `Play` / `Sports` / `Dance`) used as a feature by the ML models below.

### 4.3 Guest Self-Service ("My Events" / "My Profile")
A logged-in guest browses upcoming events not yet registered for, registers in one click, views all their own registrations (with ticket/check-in codes and status), and can cancel any registration still in `registered` status. Profile (name/email/phone) is created once and editable anytime.

### 4.4 Admin/Organizer Guest & Registration Management
Admin: full table of every guest/registration with sorting, filtering, search, and edit/delete. Organizer: the same view, filtered to their own events, read-only (editing and duplicate resolution stay Admin-only, since a guest record is shared across every organizer's events).

### 4.5 Ticket Check-In
Admin or Organizer types a guest's ticket code at the event; the system looks it up, validates it, and marks that registration attended. Organizer's version is additionally guarded to only accept codes for events they created.

### 4.6 Post-Event Review
A guest picks one of their own attended, not-yet-reviewed events from a dropdown and submits a 1-5 rating plus free-text feedback. An Admin-only testing bypass (Data Tools page) can widen this to "any event you're registered for," skipping the attended/event-over check, for testing without a real checked-in guest on hand.

## 5. Data Model

Five linked SQLite tables in a single file (`data/guest_dashboard.db`), which ships pre-seeded so the app has something to show right after a clone.

| Table | Key fields | Notes |
|---|---|---|
| **Users** | `user_id`, `username`, `password`, `role`, `display_name` | Login accounts; `role` is `admin` / `organizer` / `guest` |
| **Guests** | `guest_id`, `name`, `email`, `phone`, `user_id`, `possible_duplicate_of`, `duplicate_match_score`, `segment` | Persistent across all events; `user_id` links a guest record to its owning login account |
| **Events** | `event_id`, `event_name`, `event_date`, `start_time`, `end_time`, `tag`, `predicted_turnout`, `created_by` | `created_by` is the Admin/Organizer `user_id` that owns the event |
| **Registrations** | `registration_id`, `guest_id`, `event_id`, `registered_at`, `ticket_code`, `checked_in_at`, `attendance_status`, `predicted_no_show` | `ticket_code` is unique; `attendance_status` is `registered` / `attended` / `no_show` |
| **Reviews** | `review_id`, `guest_id`, `event_id`, `review_text`, `sentiment_score`, `rating` | Keyed on `(guest_id, event_id)`, not `registration_id` -- deleting a registration cascades to delete its review |

New columns are added via `_ensure_*_columns()` auto-migration helpers, so an existing database never needs a manual reset when the schema grows.

## 6. AI/Data Science Features

### 6.1 Guest Segmentation (Clustering)
K-Means across guests' event history (visit frequency, attendance rate) tags guests as VIP / Regular / New.

### 6.2 No-Show Prediction
Logistic regression predicting whether a registered guest will attend a given event, trained on past attendance outcomes and event tags. Trained and evaluated on the same data (no held-out set) -- a deliberate simplification for a small demo dataset.

### 6.3 Sentiment Analysis on Reviews
DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`) scores each review's free text; a sentiment trend is charted across events.

### 6.4 Attendance / Turnout Forecasting
Prophet fits a trend over past events' turnout rates, with event tag as an added regressor, to forecast expected turnout for upcoming events.

### 6.5 Duplicate Guest Detection
`difflib.SequenceMatcher` fuzzy-matches a new registration's name against existing guest records (names normalized by lowercasing and alphabetizing words) to catch the same person registering under a slightly different name/email. Flagged pairs are reviewed and resolved (merge or dismiss) by Admin.

### 6.6 Analytics Dashboard
Cross-event charts: attendance rate over time (with and without the forecast line), sentiment trend, segment breakdown. Shared rendering logic (`utils/analytics_sections.py`) powers both the system-wide Admin view and the Organizer's own-events-only view.

## 7. Tech Stack

| Layer | Tool |
|---|---|
| UI / App Framework | Streamlit (native `st.Page` + `st.navigation`) |
| Data Storage | SQLite, via Python's built-in `sqlite3` |
| ML / Modeling | scikit-learn |
| NLP | HuggingFace Transformers pipeline (DistilBERT, sentiment) |
| Forecasting | Prophet |
| Visualization | Plotly |
| Fuzzy Matching | `difflib` (standard library) |

## 8. Out of Scope

- User self-registration/signup -- accounts are pre-loaded only
- Real authentication -- passwords are plain text and shared per role; fine for a local demo, not for production
- Automated email or QR delivery -- ticket codes are shown on screen and shared manually
- Camera-based QR scanning at check-in -- typed-code validation only; a planned future addition, the data model already supports it with no rework needed
- Payment or booking system integration
- Mobile app version
- Real-time notifications
- Train/test split for the no-show model -- intentional simplification for a small demo dataset

## 9. Status

All three v2 phases -- login/roles, scoped Organizer views, and self-service Guest registration/profile/review -- are complete. Deferred for later: QR-based check-in, and a GitHub Release on the `v1.0.0` tag.