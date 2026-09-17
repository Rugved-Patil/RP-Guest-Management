"""
pages/review.py
-----------------
Guest-facing page: leave a post-event review.

Now that guests log in and My Events already shows them their own
registrations and ticket codes, this page identifies the guest from
their logged-in account (get_guest_by_user_id()) instead of asking
them to type a ticket code -- the same shift guest_events.py made away
from the old anonymous registration form.

Flow:
1. Look up this login account's guest profile. If there isn't one yet
   (first time this account has ever been used), send them to My
   Profile first -- same pattern as guest_events.py.
2. Build the list of this guest's registrations that are eligible for
   a review right now and haven't been reviewed yet:
   - Normal rule: marked "attended" AND the event's date has passed
     (see is_review_eligible() in db.py).
   - Testing bypass (Admin's Data Tools page): if it's on, any event
     the guest has registered for is fair game, regardless of
     attendance status or whether the event has happened yet.
3. Guest picks one of those from a dropdown, writes a rating + review,
   and submits. Selecting from a dropdown -- rather than a form field --
   means picking a different event updates the page immediately,
   same idea as the old ticket_code text_input reacting live.
"""

import streamlit as st
from datetime import date
from utils.db import (
    init_db,
    get_guest_by_user_id,
    get_registrations_by_guest,
    has_review,
    is_review_eligible,
    get_bypass_setting,
    add_review,
)
from utils.ml import score_sentiment
from utils.auth import require_role

init_db()
require_role("guest")

user_id = st.session_state.user_id
guest = get_guest_by_user_id(user_id)

st.title("Leave a Review")

if guest is None:
    st.info(
        "You haven't set up your guest profile yet -- head to **My Profile** "
        "and fill it in first. Once that's done, come back here to review "
        "events you've attended."
    )
    st.stop()

guest_id = guest["guest_id"]
registrations = get_registrations_by_guest(guest_id)

bypass = get_bypass_setting()
if bypass:
    st.caption(
        "⚠️ Testing bypass is ON (Data Tools page) -- any event you're "
        "registered for is open for review right now, whether or not "
        "you've been checked in or the event has happened yet."
    )

# Sort this guest's registrations into three buckets. The dropdown
# below only ever offers "reviewable" -- the other two are just used
# for the friendlier empty-state messages underneath.
already_reviewed = []
reviewable = []
not_yet_eligible = []

for r in registrations:
    if has_review(guest_id, r["event_id"]):
        already_reviewed.append(r)
    elif is_review_eligible(r, bypass=bypass):
        reviewable.append(r)
    else:
        not_yet_eligible.append(r)

if not registrations:
    st.info("You haven't registered for any events yet -- see **My Events** to browse.")
elif not reviewable:
    st.info(
        "Nothing's ready for review right now. Reviews open once you've "
        "been checked in as attended and the event is over."
    )
    if not_yet_eligible:
        st.caption(
            f"{len(not_yet_eligible)} registration(s) waiting on that -- "
            "check back after the event."
        )
else:
    def _event_label(r):
        event_date_display = date.fromisoformat(r["event_date"]).strftime("%d-%m-%Y")
        return f"{r['event_name']} -- {event_date_display}"

    # Rendered outside the form, same reason the old ticket_code
    # text_input was outside it -- so switching events updates the
    # page immediately instead of waiting for a submit click.
    reviewable_by_label = {_event_label(r): r for r in reviewable}
    selected_label = st.selectbox("Which event?", list(reviewable_by_label.keys()))
    selected = reviewable_by_label[selected_label]

    event_date_display = date.fromisoformat(selected["event_date"]).strftime("%d-%m-%Y")
    st.write(f"**Event:** {selected['event_name']} -- {event_date_display}")
    st.divider()

    with st.form("review_form"):
        # select_slider snaps to whole numbers only (no half stars),
        # and starting at 3 keeps it neutral instead of nudging guests
        # toward a 5-star default.
        rating = st.select_slider("Rating", options=[1, 2, 3, 4, 5], value=3)
        review_text = st.text_area("Your review")
        submitted = st.form_submit_button("Submit review")

        if submitted:
            if not review_text.strip():
                st.error("Please write a few words before submitting.")
            else:
                # score_sentiment() runs the text through a HuggingFace
                # model (see utils/ml.py). The very first review
                # submitted after the app starts will pause here for a
                # few seconds while that model loads -- the spinner is
                # just so the guest sees something's happening instead
                # of the page looking frozen.
                with st.spinner("Analyzing your feedback..."):
                    sentiment_score = score_sentiment(review_text.strip())
                add_review(
                    guest_id,
                    selected["event_id"],
                    review_text.strip(),
                    sentiment_score=sentiment_score,
                    rating=rating,
                )
                st.success("Thanks for your feedback!")
                st.balloons()

if already_reviewed:
    st.divider()
    st.caption(f"You've already reviewed {len(already_reviewed)} event(s) -- thanks!")
