"""
pages/guest_events.py
-----------------------
Guest-facing page: browse open events, register with one click, and
manage existing registrations -- all tied to whoever's logged in,
instead of the old anonymous form (pages/register.py, now removed)
that asked every guest to type their name/email/phone by hand every
single time.

Flow:
1. Look up this login account's guest profile via get_guest_by_user_id().
   If there isn't one yet (first time this account has ever been
   used), send them to "My Profile" to create it -- see the note down
   at the bottom of this file for why that's a separate page rather
   than a form repeated here.
2. Browse: every upcoming event they're NOT already registered for,
   with a one-click "Register" (no form fields -- we already know who
   they are).
3. My registrations: every event they ARE registered for, with their
   ticket code (this doubles as their check-in code at the door).
   Leaving a review no longer needs this code -- pages/review.py now
   looks up the guest's own attended, not-yet-reviewed events directly
   from their logged-in account instead.
4. Cancel: only offered for registrations still in "registered" status
   -- once a guest has been checked in (attended) or the event's
   passed them by (no_show), that's history, not something to cancel.
"""

import streamlit as st
from datetime import date
from utils.db import (
    init_db,
    get_guest_by_user_id,
    get_all_events,
    get_registrations_by_guest,
    add_registration,
    has_registration,
    delete_registration,
)
from utils.auth import require_role

init_db()
require_role("guest")

user_id = st.session_state.user_id
guest = get_guest_by_user_id(user_id)

st.title("My Events")

if guest is None:
    st.info(
        "You haven't set up your guest profile yet -- head to **My Profile** "
        "and fill it in once. After that, registering here is a single click."
    )
    st.stop()

guest_id = guest["guest_id"]

# --- Browse & register ----------------------------------------------------

st.subheader("Browse events")

all_events = get_all_events()
upcoming_events = [
    e for e in all_events if date.fromisoformat(e["event_date"]) >= date.today()
]

# Fetched once, up front, so both the "already registered" exclusion
# below and the "My registrations" section further down start from the
# same snapshot for this run of the page.
my_registrations = get_registrations_by_guest(guest_id)
registered_event_ids = {r["event_id"] for r in my_registrations}

available_events = [
    e for e in upcoming_events if e["event_id"] not in registered_event_ids
]

if not available_events:
    st.info("No new events open for registration right now -- check back soon.")
else:
    def _event_label(e):
        return f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}"

    event_options = {_event_label(e): e["event_id"] for e in available_events}

    with st.form("guest_register_form"):
        selected_label = st.selectbox("Which event?", list(event_options.keys()))
        submitted = st.form_submit_button("Register")

        if submitted:
            event_id = event_options[selected_label]
            # Defensive re-check -- the dropdown above already excludes
            # events this guest is registered for, but this guards
            # against a stale page (e.g. a second browser tab open to
            # this same form) trying to register twice anyway.
            if has_registration(guest_id, event_id):
                st.warning(
                    "You're already registered for this event -- see "
                    "your ticket code under 'My registrations' below."
                )
            else:
                ticket_code = add_registration(guest_id, event_id)
                st.success("You're registered! Save your ticket code below.")
                st.metric("Your ticket code", ticket_code)
                st.caption("You'll need this code for check-in at the event.")

st.divider()

# --- My registrations -------------------------------------------------
# Fetched fresh (not reusing my_registrations from above) so a
# registration made in the form just above already shows up here in
# the same run -- same "fetch the list after the form that changes it"
# pattern organizer_events.py uses for its own "existing events" list.

st.subheader("My registrations")

my_registrations = get_registrations_by_guest(guest_id)

if not my_registrations:
    st.info("You haven't registered for any events yet -- browse the list above.")
else:
    display_rows = []
    for r in my_registrations:
        display_rows.append({
            "Event": r["event_name"],
            "Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
            "Time": f"{r['start_time'] or '--'} - {r['end_time'] or '--'}",
            "Ticket / Check-in Code": r["ticket_code"],
            "Status": r["attendance_status"],
        })
    st.dataframe(display_rows, use_container_width=True)
    st.caption(
        "Your ticket code doubles as your check-in code at the door. "
        "Once you've attended and the event's over, head to Leave a "
        "Review to share feedback -- no code needed there."
    )

    # --- Cancel a registration -----------------------------------------

    st.divider()
    st.subheader("Cancel a registration")

    # Only "registered" (i.e. upcoming, not yet checked in) rows can be
    # cancelled -- an "attended" or "no_show" registration is already
    # history, not something to undo.
    cancellable = [r for r in my_registrations if r["attendance_status"] == "registered"]

    if not cancellable:
        st.caption(
            "Nothing to cancel -- only upcoming registrations that "
            "haven't been checked in yet can be cancelled."
        )
    else:
        def _reg_label(r):
            event_date_display = date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y")
            return f"{r['event_name']} -- {event_date_display} ({r['ticket_code']})"

        reg_by_label = {_reg_label(r): r for r in cancellable}
        selected_reg_label = st.selectbox(
            "Select registration to cancel", list(reg_by_label.keys())
        )
        selected_reg = reg_by_label[selected_reg_label]

        # Same two-step confirm pattern as organizer_events.py's
        # "delete an event" and admin_manage.py's "delete a
        # registration" -- a session_state flag toggled by a first
        # click, only actually deleting on a second, explicit click.
        cancel_key = f"confirm_cancel_reg_{selected_reg['registration_id']}"
        if cancel_key not in st.session_state:
            st.session_state[cancel_key] = False

        if not st.session_state[cancel_key]:
            if st.button("Cancel this registration", key=f"cancel_btn_{selected_reg['registration_id']}"):
                st.session_state[cancel_key] = True
                st.rerun()
        else:
            st.warning(
                f"This cancels your registration for {selected_reg['event_name']} "
                f"(ticket {selected_reg['ticket_code']}). This cannot be undone."
            )
            confirm_col, back_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Yes, cancel",
                    type="primary",
                    key=f"confirm_cancel_btn_{selected_reg['registration_id']}",
                ):
                    # delete_registration() also removes any review tied
                    # to this guest+event pair -- moot here since only
                    # "registered" (never-attended) rows reach this
                    # point, but it's the same shared function every
                    # other cancel/delete path in the app uses.
                    delete_registration(selected_reg["registration_id"])
                    st.session_state[cancel_key] = False
                    st.success("Registration cancelled.")
                    st.rerun()
            with back_col:
                if st.button("Never mind", key=f"back_cancel_btn_{selected_reg['registration_id']}"):
                    st.session_state[cancel_key] = False
                    st.rerun()
