"""
utils/auth.py
---------------
Login and session helpers.

This sits a layer above utils/db.py: db.py knows how to look up an
account by username, but has no idea what a "logged-in session" is --
that's what st.session_state is for, and that's what this file
manages. Unlike utils/ml.py, there's no rule against this file
importing utils/db.py; that restriction only exists to keep the ML
code framework-agnostic (see the note at the top of ml.py), which
doesn't apply here.
"""

import streamlit as st
from utils.db import get_user_by_username


def attempt_login(username, password):
    """
    Check a username/password against the users table.

    On a match, stores the account's details in st.session_state and
    returns True. On no match (wrong username OR wrong password --
    both are treated the same), session_state is left untouched and
    this returns False.
    """
    user = get_user_by_username(username.strip())
    if user is None or user["password"] != password:
        return False

    st.session_state.user_id = user["user_id"]
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.display_name = user["display_name"]
    return True


def logout():
    """Clear everything login-related out of session_state."""
    for key in ("user_id", "username", "role", "display_name"):
        st.session_state.pop(key, None)


def is_logged_in():
    """True if someone is currently logged in on this browser session."""
    return "user_id" in st.session_state


def current_role():
    """The logged-in user's role ('admin' / 'organizer' / 'guest'), or None."""
    return st.session_state.get("role")


def require_role(*allowed_roles):
    """
    Call this at the very top of any page that should only be
    reachable by certain roles, e.g. require_role("admin") or
    require_role("organizer", "admin").

    app.py only ever hands st.navigation() the pages a role is
    supposed to see in the first place, so in normal use nobody can
    even click their way to a page they're not allowed on. This
    function is the backup for that: if a page ever somehow runs
    for the wrong role (a stale bookmark from before logging out, a
    role list that gets edited wrong later, etc.), this stops the
    page's content from rendering instead of silently trusting that
    the navigation menu was the only thing keeping it hidden.

    Worth being upfront about the limits of this, in case it comes up
    later: this is access control for a local, single-user demo, not
    real security -- there's no encryption, no session expiry, and
    the "password" is compared as plain text (see DEMO_PASSWORD in
    utils/db.py). Good enough for a portfolio project; not something
    to reuse as-is for a real multi-user system.
    """
    if not is_logged_in():
        st.error("Please log in to view this page.")
        st.stop()
    if current_role() not in allowed_roles:
        st.error("You don't have access to this page.")
        st.stop()
