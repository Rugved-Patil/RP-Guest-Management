"""
pages/guest_profile.py
------------------------
Guest-facing page: create (first visit) or edit (every visit after)
the guest profile tied to this login account.

This is deliberately a separate page from "My Events" rather than an
inline form that pops up there, for two reasons:
1. It's the one place a guest's name/email/phone can be changed after
   the fact -- same "edit updates it everywhere" behavior as Admin's
   Manage Guests page (see update_guest() in utils/db.py), just
   scoped to editing your own profile instead of anyone's.
2. Keeping "browse & register" and "edit my details" on separate pages
   matches how Admin/Organizer split those same two concerns across
   different pages too, rather than mixing them into one crowded form.
"""

import streamlit as st
from utils.db import (
    init_db,
    get_guest_by_user_id,
    get_or_create_guest_for_user,
    update_guest,
)
from utils.auth import require_role

init_db()
require_role("guest")

user_id = st.session_state.user_id
guest = get_guest_by_user_id(user_id)

st.title("My Profile")

if guest is None:
    st.info(
        "Set this up once and it'll be used automatically every time "
        "you register for an event -- no re-typing your details."
    )
    with st.form("create_guest_profile_form"):
        name = st.text_input("Full name")
        email = st.text_input("Email")
        phone = st.text_input("Phone number")
        submitted = st.form_submit_button("Save profile")

        if submitted:
            if not name.strip() or not email.strip():
                st.error("Name and email are required.")
            else:
                # get_or_create_guest_for_user() handles the exact-email
                # / fuzzy-duplicate check the same way the old
                # anonymous registration form did (see get_or_create_guest()
                # in utils/db.py), then links whichever guest row it
                # returns to this login account.
                get_or_create_guest_for_user(user_id, name.strip(), email.strip(), phone.strip())
                st.success("Profile saved.")
                st.rerun()
else:
    st.caption(
        "Guests are shared across every event they've registered for, "
        "so saving changes here updates your details everywhere at "
        "once -- past registrations, reviews, and analytics all "
        "reflect the same profile."
    )
    with st.form("edit_guest_profile_form"):
        name = st.text_input("Full name", value=guest["name"])
        email = st.text_input("Email", value=guest["email"])
        phone = st.text_input("Phone number", value=guest["phone"] or "")
        submitted = st.form_submit_button("Save changes")

        if submitted:
            if not name.strip() or not email.strip():
                st.error("Name and email are required.")
            else:
                update_guest(guest["guest_id"], name.strip(), email.strip(), phone.strip())
                st.success("Profile updated.")
                st.rerun()
