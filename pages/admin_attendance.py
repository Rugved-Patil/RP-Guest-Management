"""
pages/admin_attendance.py
---------------------------
Admin-facing page: check a guest in at the door.

The organizer types in the ticket code a guest shows up with (on their
phone, printed, whatever), and we mark that registration as attended.
Typed codes only for v1 -- QR scanning is a planned v2 addition, not
built here (per the scope doc).
"""

import streamlit as st
from datetime import datetime
from utils.db import init_db, mark_attendance_by_ticket

init_db()

st.title("Check-In")
st.write("Enter the guest's ticket code and check them in.")

# Unlike register.py/review.py, this form uses clear_on_submit=True --
# check-in is something you'll do over and over in a row for a stream
# of guests at the door, so clearing the box after each one lets the
# organizer go straight into typing the next code without deleting the
# old one by hand.
with st.form("check_in_form", clear_on_submit=True):
    ticket_code = st.text_input("Ticket code").strip().upper()
    submitted = st.form_submit_button("Check in", type="primary")

    if submitted:
        if not ticket_code:
            st.error("Please enter a ticket code.")
        else:
            result = mark_attendance_by_ticket(ticket_code)

            if result["status"] == "not_found":
                st.error(
                    f"No registration found for ticket code '{ticket_code}'. "
                    "Double-check and try again."
                )

            elif result["status"] == "already_checked_in":
                reg = result["registration"]
                checked_in_display = "an earlier time"
                if reg["checked_in_at"]:
                    checked_in_display = datetime.fromisoformat(
                        reg["checked_in_at"]
                    ).strftime("%d-%m-%Y %H:%M")
                st.info(
                    f"**{reg['guest_name']}** is already checked in for "
                    f"**{reg['event_name']}** (at {checked_in_display})."
                )

            elif result["status"] == "checked_in":
                reg = result["registration"]
                st.success(f"**{reg['guest_name']}** checked in for **{reg['event_name']}**. Welcome!")
                st.balloons()
