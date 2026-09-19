"""
pages/guest_tickets.py
------------------------
Guest-facing page: "My Tickets" -- pick one of your own registrations
and see its ticket code as a QR code, ready to show at the door.

This is a display-only page: it doesn't check anyone in or change any
data. Scanning (or reading the code as text) happens on the check-in
pages (admin_attendance.py / organizer_checkin.py), which are
unaffected by anything here -- this page only calls generate_ticket_qr()
from utils/qr.py, never decode_ticket_qr().

Every registration is offered, not just upcoming ones -- if a guest
already checked in and wants to look back at an old ticket, that's
harmless; this page never marks anything attended on its own.
"""

import streamlit as st
from datetime import date
from utils.db import init_db, get_guest_by_user_id, get_registrations_by_guest
from utils.qr import generate_ticket_qr
from utils.auth import require_role

init_db()
require_role("guest")

user_id = st.session_state.user_id
guest = get_guest_by_user_id(user_id)

st.title("My Tickets")

if guest is None:
    st.info(
        "You haven't set up your guest profile yet -- head to **My Profile** "
        "and fill it in first. Once that's done, come back here to see "
        "your ticket codes."
    )
    st.stop()

guest_id = guest["guest_id"]
registrations = get_registrations_by_guest(guest_id)

if not registrations:
    st.info("You haven't registered for any events yet -- see **My Events** to browse.")
else:
    def _event_label(r):
        event_date_display = date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y")
        return f"{r['event_name']} -- {event_date_display}"

    reg_by_label = {_event_label(r): r for r in registrations}
    # Rendered outside any form, same reason as review.py's dropdown --
    # picking a different event updates the QR below immediately.
    selected_label = st.selectbox("Which event?", list(reg_by_label.keys()))
    selected = reg_by_label[selected_label]

    event_date_display = date.fromisoformat(selected["event_date"]).strftime("%d-%m-%Y")
    st.write(f"**Event:** {selected['event_name']} -- {event_date_display}")

    if selected["attendance_status"] == "attended":
        st.caption("You're already checked in for this one -- no need to show this again.")
    elif selected["attendance_status"] == "no_show":
        st.caption("This registration was marked as a no-show.")

    st.divider()

    qr_image = generate_ticket_qr(selected["ticket_code"])
    st.image(qr_image, caption="Show this to the organizer at check-in", width=250)
    st.metric("Ticket code", selected["ticket_code"])
    st.caption(
        "If the scan doesn't work for any reason, the code above works "
        "just as well typed in by hand."
    )
