import streamlit as st
from utils.db import init_db
from utils.auth import attempt_login, logout, is_logged_in, current_role

init_db()

st.set_page_config(page_title="RP Guest Management", page_icon="🎟️")

# Nobody's logged in yet: show only a login form and stop the script
# here. st.navigation() never gets called in this branch, so there's
# no sidebar menu and nothing to click into until login succeeds --
# see the note in utils/auth.py's require_role() docstring for what
# this does and doesn't protect against.
if not is_logged_in():
    st.title("RP Guest Management")
    st.caption("Log in to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

        if submitted:
            if attempt_login(username, password):
                st.rerun()  # Re-run so the rest of this file sees the new session state.
            else:
                st.error("Incorrect username or password.")

    st.stop()

# Logged in: wrap each file as a "page" Streamlit understands.
admin_events_page = st.Page("pages/admin_events.py", title="Create Event", icon="🗓️")
admin_manage_page = st.Page("pages/admin_manage.py", title="Manage Guests", icon="🗂️")
admin_attendance_page = st.Page("pages/admin_attendance.py", title="Check-In", icon="🎟️")
admin_analytics_page = st.Page("pages/admin_analytics.py", title="Analytics", icon="📊")
admin_tools_page = st.Page("pages/admin_tools.py", title="Data Tools", icon="🧹")
organizer_events_page = st.Page("pages/organizer_events.py", title="Manage My Events", icon="🗓️")
organizer_checkin_page = st.Page("pages/organizer_checkin.py", title="Check-In", icon="🎟️")
organizer_guests_page = st.Page("pages/organizer_guests.py", title="Guest List", icon="🗂️")
organizer_analytics_page = st.Page("pages/organizer_analytics.py", title="My Analytics", icon="📊")
register_page = st.Page("pages/register.py", title="Register", icon="📝")
review_page = st.Page("pages/review.py", title="Leave a Review", icon="💬")

# Which pages each role gets in their sidebar menu.
role_pages = {
    "admin": [
        admin_events_page,
        admin_manage_page,
        admin_attendance_page,
        admin_analytics_page,
        admin_tools_page,
    ],
    "organizer": [
        organizer_events_page,
        organizer_checkin_page,
        organizer_guests_page,
        organizer_analytics_page,
    ],
    "guest": [
        register_page,
        review_page,
    ],
}

pg = st.navigation(role_pages.get(current_role(), []))

# Placed after st.navigation() so it appears below the page menu in
# the sidebar, rather than above it.
with st.sidebar:
    st.divider()
    st.caption(f"Logged in as **{st.session_state.display_name}** ({current_role()})")
    if st.button("Log out"):
        logout()
        st.rerun()

pg.run()
