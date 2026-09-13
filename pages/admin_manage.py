"""
pages/admin_manage.py
-----------------------
Admin-facing page: view every guest's registrations in one table, with
filtering by event and attendance status, plus a free-text search on
name/email. Below that, pick any one registration to edit the guest's
details or delete the registration outright. Also surfaces guests
flagged as possible duplicates (see utils/db.py's
find_possible_duplicate()) with Merge / Dismiss actions.

One row = one registration (a guest + event pairing), not one row per
guest, in the main table below. Ticket code and attendance status
belong to the registration, not the guest, so a guest registered for
3 events shows up as 3 rows here -- that's what makes "guests
registered for a particular event" and "ticket code visible" both
possible at once. The same is true of the edit/delete section: editing
a guest's details there affects them everywhere, since guests are
shared across events, but deleting only removes the one selected
registration.
"""

import streamlit as st
from datetime import date, datetime
from utils.db import (
    init_db,
    get_all_registrations_detailed,
    get_all_events,
    get_possible_duplicates,
    dismiss_duplicate_flag,
    merge_guests,
    update_guest,
    delete_registration,
)
from utils.auth import require_role

init_db()
require_role("admin")

st.title("Manage Guests")

registrations = get_all_registrations_detailed()

if not registrations:
    st.info("No registrations yet.")
    st.stop()

# --- Build filter options -------------------------------------------

events = get_all_events()

def _event_label(e):
    return f"{e['event_name']} -- {date.fromisoformat(e['event_date']).strftime('%d-%m-%Y')}"

event_id_by_label = {_event_label(e): e["event_id"] for e in events}
event_options = ["All events"] + list(event_id_by_label.keys())

filter_col, status_col, search_col = st.columns([2, 1, 2])
with filter_col:
    selected_event_label = st.selectbox("Filter by event", event_options)
with status_col:
    status_filter = st.selectbox(
        "Attendance status", ["All", "registered", "attended", "no_show"]
    )
with search_col:
    search_text = st.text_input("Search by name or email").strip().lower()

selected_event_id = event_id_by_label.get(selected_event_label)  # None means "All events"

# --- Apply filters -----------------------------------------------------
# Filtering a plain Python list here rather than writing separate SQL
# queries per filter combo -- the dataset is small (a few hundred rows
# at most for a portfolio demo), so this is simpler to read and just as
# fast in practice.

filtered = registrations

if selected_event_id is not None:
    filtered = [r for r in filtered if r["event_id"] == selected_event_id]

if status_filter != "All":
    filtered = [r for r in filtered if r["attendance_status"] == status_filter]

if search_text:
    filtered = [
        r for r in filtered
        if search_text in r["guest_name"].lower() or search_text in r["email"].lower()
    ]

st.caption(f"Showing {len(filtered)} of {len(registrations)} registrations.")

# --- Build the table ---------------------------------------------------

display_rows = []
for r in filtered:
    checked_in_display = "--"
    if r["checked_in_at"]:
        checked_in_display = datetime.fromisoformat(r["checked_in_at"]).strftime("%d-%m-%Y %H:%M")

    duplicate_display = "--"
    if r["possible_duplicate_of"] is not None:
        duplicate_display = f"Possible match: guest #{r['possible_duplicate_of']} ({r['duplicate_match_score']}%)"

    display_rows.append({
        "Guest": r["guest_name"],
        "Email": r["email"],
        "Phone": r["phone"] or "--",
        "Segment": r["segment"] or "Not yet segmented",
        "Event": r["event_name"],
        "Event Date": date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y"),
        "Ticket Code": r["ticket_code"],
        "Attendance": r["attendance_status"],
        "Checked In": checked_in_display,
        "Possible Duplicate": duplicate_display,
    })

# st.dataframe supports sorting by clicking a column header out of the
# box -- no extra sorting code needed here for that.
st.dataframe(display_rows, use_container_width=True)

st.divider()

# --- Edit or delete a registration --------------------------------------

st.subheader("Edit or delete a registration")

if not filtered:
    st.caption("No registrations match the filters above to edit or delete.")
else:
    def _registration_label(r):
        return f"{r['guest_name']} -- {r['event_name']} ({r['ticket_code']})"

    registration_by_label = {_registration_label(r): r for r in filtered}
    selected_label = st.selectbox(
        "Select a registration", list(registration_by_label.keys())
    )
    selected = registration_by_label[selected_label]

    edit_col, delete_col = st.columns(2)

    with edit_col:
        st.markdown("**Edit guest details**")
        st.caption(
            "Guests are shared across every event they're registered "
            "for, so this updates their name/email/phone everywhere -- "
            "not just on this one registration."
        )
        with st.form(f"edit_guest_form_{selected['guest_id']}"):
            edit_name = st.text_input("Name", value=selected["guest_name"])
            edit_email = st.text_input("Email", value=selected["email"])
            edit_phone = st.text_input("Phone", value=selected["phone"] or "")
            save_clicked = st.form_submit_button("Save changes")

            if save_clicked:
                if not edit_name.strip() or not edit_email.strip():
                    st.error("Name and email are required.")
                else:
                    update_guest(
                        selected["guest_id"],
                        edit_name.strip(),
                        edit_email.strip(),
                        edit_phone.strip(),
                    )
                    st.success(f"Updated {edit_name.strip()}'s details.")
                    st.rerun()

    with delete_col:
        st.markdown("**Delete this registration**")
        st.caption(
            "Removes just this one guest + event registration (and its "
            "review, if any) -- not the guest's other registrations, "
            "and not the guest record itself."
        )
        delete_key = f"confirm_delete_reg_{selected['registration_id']}"
        if delete_key not in st.session_state:
            st.session_state[delete_key] = False

        if not st.session_state[delete_key]:
            if st.button("Delete registration", key=f"delete_btn_{selected['registration_id']}"):
                st.session_state[delete_key] = True
                st.rerun()
        else:
            st.warning(
                f"This permanently deletes {selected['guest_name']}'s "
                f"registration for {selected['event_name']} "
                f"(ticket {selected['ticket_code']}). This cannot be undone."
            )
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button(
                    "Yes, delete",
                    key=f"confirm_delete_btn_{selected['registration_id']}",
                    type="primary",
                ):
                    delete_registration(selected["registration_id"])
                    st.session_state[delete_key] = False
                    st.success("Registration deleted.")
                    st.rerun()
            with cancel_col:
                if st.button("Cancel", key=f"cancel_delete_btn_{selected['registration_id']}"):
                    st.session_state[delete_key] = False
                    st.rerun()

st.divider()

# --- Possible duplicate guests -----------------------------------------

duplicate_pairs = get_possible_duplicates()

st.subheader("Possible duplicate guests")

if not duplicate_pairs:
    st.caption("No possible duplicates flagged right now.")
else:
    st.caption(
        f"{len(duplicate_pairs)} guest(s) flagged as a likely match for an "
        "existing guest at registration time. Review each pair below."
    )
    for pair in duplicate_pairs:
        flagged_id = pair["flagged_id"]
        confirm_key = f"confirm_merge_{flagged_id}"
        if confirm_key not in st.session_state:
            st.session_state[confirm_key] = False

        with st.container(border=True):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**New registration:** {pair['flagged_name']}")
                st.caption(
                    f"{pair['flagged_email']} · {pair['flagged_phone'] or '--'} · "
                    f"{pair['flagged_registrations']} registration(s)"
                )
            with col_b:
                st.markdown(f"**Existing guest:** {pair['original_name']} (#{pair['original_id']})")
                st.caption(
                    f"{pair['original_email']} · {pair['original_phone'] or '--'} · "
                    f"{pair['original_registrations']} registration(s)"
                )
            st.caption(f"Name match score: {pair['duplicate_match_score']}%")

            if not st.session_state[confirm_key]:
                merge_col, dismiss_col = st.columns(2)
                with merge_col:
                    if st.button("Merge into existing guest", key=f"merge_btn_{flagged_id}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
                with dismiss_col:
                    if st.button("Dismiss (not a duplicate)", key=f"dismiss_btn_{flagged_id}"):
                        dismiss_duplicate_flag(flagged_id)
                        st.success(f"Dismissed -- {pair['flagged_name']} is no longer flagged.")
                        st.rerun()
            else:
                st.warning(
                    f"This moves all of {pair['flagged_name']}'s registrations and "
                    f"reviews onto {pair['original_name']} (#{pair['original_id']}), "
                    "then deletes this guest record. This cannot be undone."
                )
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    if st.button("Yes, merge", key=f"confirm_merge_btn_{flagged_id}", type="primary"):
                        result = merge_guests(flagged_id, pair["original_id"])
                        st.session_state[confirm_key] = False
                        st.success(
                            f"Merged. Moved {result['moved_registrations']} "
                            f"registration(s) ({result['resolved_overlaps']} overlapping "
                            f"event(s) auto-resolved) and {result['moved_reviews']} "
                            f"review(s) onto {pair['original_name']}."
                        )
                        st.rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_merge_btn_{flagged_id}"):
                        st.session_state[confirm_key] = False
                        st.rerun()