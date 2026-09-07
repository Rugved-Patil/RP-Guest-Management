import streamlit as st

# Wrap each file as a "page" Streamlit understands
admin_events_page = st.Page("pages/admin_events.py", title="Create Event", icon="🗓️")
register_page = st.Page("pages/register.py", title="Register", icon="📝")
review_page = st.Page("pages/review.py", title="Leave a Review", icon="💬")
admin_manage_page = st.Page("pages/admin_manage.py", title="Manage Guests", icon="🗂️")
admin_attendance_page = st.Page("pages/admin_attendance.py", title="Check-In", icon="🎟️")
admin_analytics_page = st.Page("pages/admin_analytics.py", title="Analytics", icon="📊")

# Build the sidebar menu from those pages
pg = st.navigation([
    admin_events_page,
    register_page,
    review_page,
    admin_manage_page,
    admin_attendance_page,
    admin_analytics_page,
])

# Actually run whichever page is selected
pg.run()
