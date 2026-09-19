"""
utils/qr.py
------------
QR code generation and decoding for ticket check-in.

A QR code here is nothing more than a small image encoding the same
5-character ticket code (e.g. "LPT43") that already exists in plain
text -- no new data, no new database column. Two functions:

- generate_ticket_qr(): guest side. Turns a ticket code into a PNG
  image, generated fresh each call rather than stored anywhere -- same
  as the ticket code itself isn't pre-rendered until something asks
  for it.
- decode_ticket_qr(): check-in side. Reads a ticket code back out of a
  photo (e.g. from st.camera_input()). Used by admin_attendance.py and
  organizer_checkin.py as a *second* way to get a ticket code string,
  alongside the existing typed-code text_input -- both paths end up
  calling the exact same mark_attendance_by_ticket() in db.py, so
  nothing about how check-in actually works had to change, only how
  the code gets in.

This file doesn't import utils/db.py or utils/ml.py, and neither of
those import this one -- same "keep each concern in its own file, no
circular imports" pattern the project already follows for ml.py.
"""

import io

import cv2
import numpy as np
import qrcode


def generate_ticket_qr(ticket_code):
    """
    Build a QR code image encoding just the ticket code itself.
    Returns PNG image bytes, ready to hand straight to st.image().
    """
    img = qrcode.make(ticket_code)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def decode_ticket_qr(image_bytes):
    """
    Read a ticket code back out of a photo. image_bytes is the raw
    bytes of a photo (e.g. st.camera_input()'s .getvalue()) -- any
    common image format (JPEG, PNG) works, since OpenCV sniffs the
    format itself rather than needing to be told.

    Returns the decoded text (stripped, uppercased -- matching how
    typed ticket codes are normalized elsewhere), or None if no QR
    code was found in the photo. Callers should treat None the same
    as an empty typed ticket-code box: nothing to look up yet.

    cv2.QRCodeDetector() is OpenCV's own built-in decoder -- no extra
    system library needed beyond the opencv-python-headless pip
    package itself, unlike zbar-based alternatives (pyzbar), which
    need a separate OS-level install (e.g. `brew install zbar`).
    """
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        # Not a decodable image at all (corrupt bytes, empty capture).
        return None

    detector = cv2.QRCodeDetector()
    decoded_text, points, _ = detector.detectAndDecode(image)

    if not decoded_text:
        # A QR code either wasn't found in the frame, or was found but
        # was unreadable (too blurry, too small, bad angle).
        return None

    return decoded_text.strip().upper()
