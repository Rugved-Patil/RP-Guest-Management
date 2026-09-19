# RP-Guest-Management

A Streamlit dashboard for managing guests across recurring events, built to practice applying machine learning to a real, structured workflow instead of a toy dataset. It covers the full lifecycle of an event -- creation, ticketed registration, check-in, and post-event feedback -- behind a role-based login, and layers in a handful of ML features on top: clustering, prediction, sentiment analysis, and forecasting.

For the full feature scope, data model, and what's explicitly out of scope, see [`RP-Guest_Management_Scope.md`](./RP-Guest_Management_Scope.md). This README is the shorter "how do I run this" version.

## What it does

Three roles share one app: **Admin** has full, unscoped access to everything; **Organizer** gets the same day-to-day tools (create events, check guests in, view guests, see analytics) scoped to only the events they created; **Guest** logs in to browse and register for events, manage their own registrations, and leave reviews -- no typing in codes or re-entering their details every time.

Flow, step by step:

1. An Organizer (or Admin) creates an event
2. A logged-in Guest registers with one click from **My Events** and receives a unique ticket code on screen
3. The registration gets checked against existing guest profiles for possible duplicates
4. A model flags guests who are likely to no-show
5. Admin/Organizer can search, view, and edit guests and registrations at any time
6. At the door, the ticket code is typed in to check the guest in
7. Once a guest is marked attended and the event is over, **Leave a Review** shows it in their dropdown of reviewable events
8. That review gets scored for sentiment automatically
9. Everything feeds into the analytics dashboard -- system-wide for Admin, scoped to their own events for Organizer

## Features

Core stuff:
- Login with role-based navigation (Admin / Organizer / Guest)
- Event creation, owned by whoever created it (`events.created_by`)
- Guest self-service: browse open events, one-click register, view/cancel registrations, edit profile -- all tied to the logged-in account
- Admin: full search, view, and edit for every guest and registration; Organizer: read-only guest list scoped to their own events
- Check-in by typing a ticket code, or by photographing the guest's QR code (Admin: any event; Organizer: their own events only) -- both feed the same validation, so typing always works as a fallback if a scan doesn't
- Guests can view/show their QR code anytime on **My Tickets**, per registration
- Post-event reviews: a logged-in guest picks from their own attended, not-yet-reviewed events in a dropdown -- no code to type
- Fuzzy duplicate guest detection, with merge and dismiss options (Admin-only)

ML/DS features:
- Sentiment analysis on reviews, using DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- No-show prediction with logistic regression, trained on guest history and event tags
- Guest segmentation with K-Means, sorting guests into VIP, Regular, or New based on how often they come and how reliably they show up
- Turnout forecasting with Prophet, predicting expected attendance rate for upcoming events, using event tag as a regressor
- A plain attendance-rate-over-time chart, showing the same historical numbers the forecast trains on without the forecast line mixed in

## Tech stack

Streamlit for the UI (native `st.Page` and `st.navigation`), SQLite for storage, scikit-learn for modeling, HuggingFace Transformers for sentiment, Prophet for forecasting, Plotly for charts, `qrcode` for generating ticket QR codes, and `opencv-python-headless` for reading them back out of a photo. Pinned versions: `transformers==5.16.1`, `torch==2.14.0`, `plotly==7.0.0`, `scikit-learn==1.9.0`, `prophet==1.4.0`, `qrcode==8.2`, `opencv-python-headless==4.13.0.92`.

## Project structure

```
RP-Guest-Management/
├── app.py                       # entry point: login gate + role-based navigation
├── pages/
│   ├── admin_events.py          # admin: create events
│   ├── admin_manage.py          # admin: view/search/edit guests, duplicate merge/dismiss
│   ├── admin_attendance.py      # admin: ticket check-in -- type a code, or scan a QR photo
│   ├── admin_analytics.py       # admin: sentiment, no-show, segmentation, forecasting (system-wide)
│   ├── admin_tools.py           # admin: sample data seeder, reset, review-eligibility testing bypass
│   ├── organizer_events.py      # organizer: manage my events (create/edit/delete, own events only)
│   ├── organizer_checkin.py     # organizer: ticket check-in (type or scan), guarded to own events
│   ├── organizer_guests.py      # organizer: read-only guest list, scoped to own events
│   ├── organizer_analytics.py   # organizer: same analytics as admin, scoped to own events
│   ├── guest_events.py          # guest: browse/register/cancel, view ticket codes ("My Events")
│   ├── guest_profile.py         # guest: create-once/edit-anytime profile ("My Profile")
│   ├── guest_tickets.py         # guest: view a registration's ticket as a QR code ("My Tickets")
│   └── review.py                # guest: pick an attended, not-yet-reviewed event and leave feedback
├── data/
│   └── guest_dashboard.db       # SQLite database, ships pre-seeded
├── utils/
│   ├── db.py                    # all data-access functions, schema, migrations
│   ├── auth.py                  # login/session helpers, role-gating
│   ├── ml.py                    # clustering, prediction, sentiment, forecasting logic
│   ├── analytics_sections.py    # shared chart/table rendering, reused by admin + organizer analytics
│   └── qr.py                    # generate a ticket's QR code, and decode one back from a photo
├── RP-Guest_Management_Scope.md # full project scope: features, data model, out-of-scope
└── requirements.txt
```

`utils/ml.py` never imports `utils/db.py`, to avoid a circular import. ML functions just take data in as plain arguments and hand back results, so there's no dependency loop. `db.py` is the only file that imports `ml.py`. `utils/auth.py` sits above `db.py` and provides login/session logic to every page.

## Login

Three roles, 33 pre-loaded accounts total: **Admin** (full access), **Organizer** (scoped event management -- create, check in, view guests, analytics, all limited to events they created), and **Guest** (browse/register/cancel events, manage their profile, leave reviews). Accounts are pre-loaded into the database automatically the first time the app runs -- there's no signup form. Every account shares the same demo password.

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `password123` |
| Organizer | `organizer1` ... `organizer7` | `password123` |
| Guest | `guest1` ... `guest25` | `password123` |

This is plain-text, shared-password auth, which is fine for a local demo but not how a real system would do it -- see Known limitations below.

## Data model

Five linked SQLite tables:
- **Users**: login accounts for the role system (`user_id`, `username`, `password`, `role`, `display_name`)
- **Guests**: persistent guest profiles, reused across events -- `user_id` optionally links a guest record to the login account that owns it
- **Events**: one row per event, `created_by` links it to the Admin/Organizer account that created it
- **Registrations**: links a guest to a specific event, holds the ticket code, check-in time, attendance status, and no-show prediction
- **Reviews**: tied to a specific guest + event pair, only reachable once that registration is marked attended and the event is over (or the Data Tools testing bypass is on)

## Setup

1. Clone the repo and open it in your editor.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   Prophet depends on cmdstanpy, which compiles a small Stan model the first time it runs. On Windows you'll need the "Desktop development with C++" workload from Visual Studio Build Tools installed first, or that first run can fail.
4. Run it:
   ```
   streamlit run app.py
   ```

The database gets created and migrated automatically the first time you run the app. If you'd rather not enter everything by hand, log in as `admin` and use the Data Tools page's sample data seeder to fill it with realistic demo data.

The **Scan QR** tab on Check-In uses your browser's webcam, which browsers only allow over HTTPS or on `localhost` -- running locally with `streamlit run app.py` is fine (localhost counts), but this would need HTTPS if it were ever deployed elsewhere over plain HTTP. Your browser will prompt for camera permission the first time you open that tab.

## Known limitations

A few things are left out on purpose, not things that got missed:
- Passwords are plain text and shared across every account of a given role -- fine for a local demo, not something a real system would ever do
- No signup flow -- accounts are pre-loaded only, matching the v2 plan's "pre-loaded accounts, no self-registration" decision
- Role-gating relies on which pages get handed to `st.navigation()` for the current role, plus a same-purpose check at the top of each page (see `require_role()` in `utils/auth.py`) -- not real security, just enough to keep each role looking at the right thing in a local demo
- No automated email or QR delivery, ticket codes are shown on screen and shared manually
- QR scanning is snapshot-based (`st.camera_input()` -- one photo per click, decoded automatically), not a continuous live scanner like a checkout scanner; typing a code stays available as a fallback if a scan doesn't read
- No payment or booking integration
- The no-show model trains and gets evaluated on the same data, with no train/test split. That's a deliberate simplification for a small demo dataset, not meant to be a rigorous evaluation.

## Roadmap

Done: login shell with role-gated navigation, event ownership with scoped Organizer views, a fully self-service Guest experience (browse/register/cancel, profile management, and picking events to review by logged-in identity instead of typing a ticket code), and snapshot-based QR check-in alongside the original typed-code flow.

Next up: whenever convenient, a GitHub Release on top of the `v1.0.0` tag is still sitting there, ready to publish. A live continuous-scan check-in (rather than photo-per-click) is a possible future upgrade if the snapshot flow ever feels clunky in practice, but isn't planned yet.
