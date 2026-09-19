import streamlit as st
from utils.db import init_db
from utils.auth import attempt_login, logout, is_logged_in, current_role

init_db()

st.set_page_config(page_title="RP Guest Management", page_icon="🎟️")


def render_login():
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


# Nobody's logged in yet: route to a single "page" that's really just
# the login form above, instead of skipping st.navigation() entirely.
#
# This matters more than it looks like it should. Per Streamlit's own
# docs: "As soon as any session of your app executes the st.navigation
# command, your app will ignore the pages/ directory." In other words,
# st.navigation() isn't just "the thing that draws the menu" -- calling
# it at all is what tells Streamlit to stop auto-listing every file
# under pages/ in the sidebar. The earlier version of this file only
# called st.navigation() in the "logged in" branch below, and showed
# the login form + st.stop() before ever reaching it -- so on every run
# where nobody was logged in (first load, and right after logging out),
# Streamlit had never been told to stop its own default behavior, and
# fell back to listing every single page it could find in pages/,
# regardless of role or login state.
#
# Wrapping the login form as its own st.Page and always calling
# st.navigation() -- logged in or not -- means Streamlit is under
# explicit control on every single run, so that fallback never
# triggers. position="hidden" is redundant here in current Streamlit
# versions (a single-page list already hides the nav widget on its
# own), but it's left in on purpose to say clearly, in the code, that
# no menu should ever show on this screen.
if not is_logged_in():
    login_page = st.Page(render_login, title="Log In")
    pg = st.navigation([login_page], position="hidden")
    pg.run()
    st.stop()

# Logged in: wrap each file as a "page" Streamlit understands.
admin_events_page = st.Page("pages/admin_events.py", title="Create Event", icon="🗓️")
admin_manage_page = st.Page("pages/admin_manage.py", title="Manage Guests", icon="🗂️")
admin_attendance_page = st.Page("pages/admin_attendance.py", title="Check-In", icon="🎟️")
admin_analytics_page = st.Page("pages/admin_analytics.py", title="Analytics", icon="📊")
admin_tools_page = st.Page("pages/admin_tools.py", title="Admin Tools", icon="🧹")
organizer_events_page = st.Page("pages/organizer_events.py", title="Manage My Events", icon="🗓️")
organizer_checkin_page = st.Page("pages/organizer_checkin.py", title="Check-In", icon="🎟️")
organizer_guests_page = st.Page("pages/organizer_guests.py", title="Guest List", icon="🗂️")
organizer_analytics_page = st.Page("pages/organizer_analytics.py", title="My Analytics", icon="📊")
guest_events_page = st.Page("pages/guest_events.py", title="My Events", icon="🎟️")
guest_profile_page = st.Page("pages/guest_profile.py", title="My Profile", icon="👤")
guest_tickets_page = st.Page("pages/guest_tickets.py", title="My Tickets", icon="🎫")
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
        guest_events_page,
        guest_profile_page,
        guest_tickets_page,
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
