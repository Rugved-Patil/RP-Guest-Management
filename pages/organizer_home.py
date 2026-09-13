"""
pages/organizer_home.py
-------------------------
Placeholder landing page for the Organizer role.

Real Organizer-specific views -- their own events, check-in scoped to
those events, and analytics scoped to those events -- are the next
step in the v2 plan, built once events.created_by exists in the data
model. For now this just confirms the login + role-gating shell works
correctly for an Organizer account.
"""

import streamlit as st
from utils.auth import require_role

require_role("organizer")

st.title("Organizer Dashboard")
st.info(
    "This is a placeholder. Once event ownership is added to the data "
    "model, this page becomes your view of the events you created -- "
    "check-in, guest management, and analytics scoped to just those "
    "events, instead of everything across the whole system."
)
