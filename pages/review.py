"""
pages/review.py
-----------------
Guest-facing page: leave a post-event review.

There's no login system yet, so the ticket code the guest got at
registration doubles as their identity here -- typing it in is how we
figure out which guest + which event a review belongs to. Same idea as
the check-in page (admin_attendance.py) uses.

Flow:
1. Guest types in their ticket code.
2. We look up the matching registration (guest + event details).
3. We check whether they're allowed to review yet: marked "attended"
   AND the event date has passed. (Admin can flip a testing bypass on
   the Data Tools page to skip this check, for testing without a real
   checked-in guest on hand.)
4. If eligible and they haven't already reviewed this event, show the
   rating + text form.
"""

import streamlit as st
from datetime import date
from utils.db import (
    init_db,
    get_registration_by_ticket,
    has_review,
    is_review_eligible,
    get_bypass_setting,
    add_review,
)
from utils.ml import score_sentiment
from utils.auth import require_role

init_db()
require_role("guest")

st.title("Leave a Review")
st.write("Enter the ticket code you received when you registered.")

# A plain text_input (not inside a form) reruns the page the moment the
# guest types something, so the lookup below happens live -- no extra
# "search" button needed for something this simple.
ticket_code = st.text_input("Ticket code").strip().upper()

if ticket_code:
    registration = get_registration_by_ticket(ticket_code)

    if registration is None:
        st.error("We couldn't find a registration with that ticket code. Double-check and try again.")
    else:
        event_date_display = date.fromisoformat(registration["event_date"]).strftime("%d-%m-%Y")
        st.write(f"**Event:** {registration['event_name']} -- {event_date_display}")
        st.write(f"**Registered as:** {registration['guest_name']}")
        st.divider()

        if has_review(registration["guest_id"], registration["event_id"]):
            st.info("You've already submitted a review for this event. Thanks for your feedback!")
        else:
            bypass = get_bypass_setting()
            if bypass:
                st.caption(
                    "Testing bypass is ON (Data Tools page) -- the attended / "
                    "event-over check is being skipped."
                )

            if is_review_eligible(registration, bypass=bypass):
                with st.form("review_form"):
                    # select_slider snaps to whole numbers only (no half
                    # stars), and starting at 3 keeps it neutral instead
                    # of nudging guests toward a 5-star default.
                    rating = st.select_slider(
                        "Rating", options=[1, 2, 3, 4, 5], value=3
                    )
                    review_text = st.text_area("Your review")
                    submitted = st.form_submit_button("Submit review")

                    if submitted:
                        if not review_text.strip():
                            st.error("Please write a few words before submitting.")
                        else:
                            # score_sentiment() runs the text through a
                            # HuggingFace model (see utils/ml.py). The
                            # very first review submitted after the app
                            # starts will pause here for a few seconds
                            # while that model loads -- the spinner is
                            # just so the guest sees something's
                            # happening instead of the page looking frozen.
                            with st.spinner("Analyzing your feedback..."):
                                sentiment_score = score_sentiment(review_text.strip())
                            add_review(
                                registration["guest_id"],
                                registration["event_id"],
                                review_text.strip(),
                                sentiment_score=sentiment_score,
                                rating=rating,
                            )
                            st.success("Thanks for your feedback!")
                            st.balloons()
            elif registration["attendance_status"] != "attended":
                st.warning(
                    "Reviews are only open to guests who were checked in as "
                    "attended at the event."
                )
            else:
                st.warning(
                    f"Reviews open once the event is over. Come back after "
                    f"{event_date_display}."
                )