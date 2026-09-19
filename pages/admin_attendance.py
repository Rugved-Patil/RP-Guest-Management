"""
pages/admin_attendance.py
---------------------------
Admin-facing page: check a guest in at the door.

Two ways to get a ticket code in: type it, or scan a QR code -- added
alongside typed entry, not replacing it. A blurry photo or a dim phone
screen is a real way a scan can fail at a live door, so typing stays
as the reliable fallback (see pages/guest_tickets.py for where the
guest's QR code comes from). Both tabs below end up calling the exact
same mark_attendance_by_ticket(), so the check-in logic itself is
unchanged from before QR scanning existed -- only how the ticket code
gets in differs.
"""

import streamlit as st
from datetime import datetime
from utils.db import init_db, mark_attendance_by_ticket
from utils.qr import decode_ticket_qr
from utils.auth import require_role

init_db()
require_role("admin")

st.title("Check-In")
st.write("Enter the guest's ticket code and check them in.")


def _show_checkin_result(ticket_code, result):
    """
    Shared by both tabs below -- same result dict shape either way,
    since both call the same mark_attendance_by_ticket().
    """
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


type_tab, scan_tab = st.tabs(["Type code", "Scan QR"])

with type_tab:
    # Unlike the review form (pages/review.py), this one uses clear_on_submit=True --
    # check-in happens over and over in a row for a stream of guests at
    # the door, so clearing the box after each one lets the organizer go
    # straight into typing the next code without deleting the old one by
    # hand.
    with st.form("check_in_form", clear_on_submit=True):
        ticket_code = st.text_input("Ticket code").strip().upper()
        submitted = st.form_submit_button("Check in", type="primary")

        if submitted:
            if not ticket_code:
                st.error("Please enter a ticket code.")
            else:
                _show_checkin_result(ticket_code, mark_attendance_by_ticket(ticket_code))

with scan_tab:
    st.caption(
        "Photograph the guest's QR code (from their My Tickets page) -- "
        "check-in happens automatically once it's read. Use the retake "
        "icon on the photo before scanning the next guest."
    )
    photo = st.camera_input("Scan the guest's QR code")

    if photo is not None:
        ticket_code = decode_ticket_qr(photo.getvalue())

        if ticket_code is None:
            st.warning(
                "Couldn't read a QR code in that photo -- try again with "
                "better lighting or a steadier angle, or use the "
                "**Type code** tab instead."
            )
        else:
            st.caption(f"Read ticket code: **{ticket_code}**")
            _show_checkin_result(ticket_code, mark_attendance_by_ticket(ticket_code))
