# RP-Guest-Management

A Streamlit dashboard for managing guests across recurring events, built to practice applying machine learning to a real, structured workflow instead of a toy dataset. It covers the full lifecycle of an event: registration, ticketing, check-in, and post-event feedback, and layers in a handful of ML features on top: clustering, prediction, sentiment analysis, and forecasting.

## What it does

An organizer creates an event. A guest registers and gets a unique ticket code on screen. At the door, the organizer types that code in to check the guest in. Afterward, guests who actually showed up can leave a review, and that review gets scored for sentiment automatically. All of it rolls up into an analytics page that also predicts no-shows before the event, groups guests into loyalty tiers, flags likely duplicate profiles, and forecasts how many people will actually turn up to upcoming events.

Flow, step by step:

1. Organizer creates an event
2. Guest registers and receives a ticket code
3. The new registration gets checked against existing guests for possible duplicates
4. A model flags guests who are likely to no-show
5. Organizer can search, view, and edit guests and registrations any time
6. At the event, the organizer types in the guest's code to check them in
7. Guests who attended can leave a review afterward
8. That review gets scored for sentiment
9. Everything feeds into the analytics dashboard

## Features

Core stuff:
- Event creation
- Guest registration with generated ticket codes
- Admin search, view, and edit for guests and registrations
- Check-in by typing a ticket code
- Post-event reviews, only available to guests who actually attended
- Fuzzy duplicate guest detection, with merge and dismiss options

ML/DS features:
- Sentiment analysis on reviews, using DistilBERT (`distilbert-base-uncased-finetuned-sst-2-english`)
- No-show prediction with logistic regression, trained on guest history and event tags
- Guest segmentation with K-Means, sorting guests into VIP, Regular, or New based on how often they come and how reliably they show up
- Turnout forecasting with Prophet, predicting expected attendance rate for upcoming events, using event tag as a regressor
- A plain attendance-rate-over-time chart, showing the same historical numbers the forecast trains on without the forecast line mixed in

## Tech stack

Streamlit for the UI (native `st.Page` and `st.navigation`), SQLite for storage, scikit-learn for modeling, HuggingFace Transformers for sentiment, Prophet for forecasting, and Plotly for charts.

## Project structure

```
RP-Guest-Management/
├── app.py                    # entry point, sets up navigation
├── pages/
│   ├── admin_events.py       # admin: create events
│   ├── register.py           # guest facing: registration form
│   ├── review.py             # guest facing: post-event review form
│   ├── admin_manage.py       # admin: view/search/edit guests, duplicate merge/dismiss
│   ├── admin_attendance.py   # admin: ticket check-in
│   ├── admin_analytics.py    # admin: sentiment, no-show, segmentation, forecasting
│   └── admin_tools.py        # admin: sample data seeder and reset
├── data/
│   └── guest_dashboard.db    # SQLite database
├── utils/
│   ├── db.py                 # all data-access functions
│   └── ml.py                 # clustering, prediction, sentiment, forecasting logic
└── requirements.txt
```

`utils/ml.py` never imports `utils/db.py`, to avoid a circular import. ML functions just take data in as plain arguments and hand back results, so there's no dependency loop. `db.py` is the only file that imports `ml.py`.

## Data model

Four linked SQLite tables:
- **Guests**: persistent guest profiles, reused across events
- **Events**: one row per event
- **Registrations**: links a guest to a specific event, holds the ticket code, check-in time, attendance status, and no-show prediction
- **Reviews**: tied to a specific registration, only exists for guests marked attended

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

The database gets created and migrated automatically the first time you run the app. If you'd rather not enter everything by hand, the Data Tools page has a sample data seeder that fills it with realistic demo data.

## Known limitations (v1)

A few things are left out on purpose for this version, not things that got missed:
- No login or user accounts, the app is single-session with no roles
- No authentication or access control
- No automated email or QR delivery, ticket codes are shown on screen and shared manually
- Check-in is typed-code only, no camera-based QR scanning
- No payment or booking integration
- The no-show model trains and gets evaluated on the same data, with no train/test split. That's a deliberate simplification for a small demo dataset, not meant to be a rigorous evaluation.

## Roadmap

Next up is a v2 branch that adds a login system with three access levels (Admin, Organizer, Guest), scopes event management and analytics to whichever organizer created the event, and turns guest registration into a logged-in, self-service flow. QR-based check-in might come after that.