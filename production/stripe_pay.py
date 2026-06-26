"""
Minimal Stripe Checkout helpers (requests only, no SDK).

Hosted Checkout: we create a Checkout Session server-side and redirect the user to Stripe's page,
so no card data ever touches us. On success Stripe redirects back with the session id, which we
retrieve and confirm `payment_status == "paid"` before unlocking the video.

Reads STRIPE_SECRET_KEY from env or .env. If unset, enabled() is False and landing.py keeps the mock.
"""
import os

import requests

import config

API = "https://api.stripe.com/v1"
PRICE_CENTS = 500   # $5.00


def _env(name):
    v = os.environ.get(name)
    envf = config.ROOT / ".env"
    if not v and envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(name + "="):
                v = line.split("=", 1)[1].strip()
    return v


def _key():
    return _env("STRIPE_SECRET_KEY")


def publishable():
    return _env("STRIPE_PUBLISHABLE_KEY")


def enabled():
    return bool(_key())


def create_intent():
    """Create a $5 PaymentIntent for the inline Apple Pay / Google Pay (Express Checkout) flow."""
    data = {"amount": str(PRICE_CENTS), "currency": "usd",
            "automatic_payment_methods[enabled]": "true"}
    r = requests.post(f"{API}/payment_intents", auth=(_key(), ""), data=data, timeout=20)
    r.raise_for_status()
    return r.json()["client_secret"]


def intent_paid(pi):
    """True if a PaymentIntent succeeded."""
    r = requests.get(f"{API}/payment_intents/{pi}", auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    return r.json().get("status") == "succeeded"


def create_session(jid, base_url):
    """Create a one-time $5 Checkout Session, return (checkout_url, session_id)."""
    data = {
        "mode": "payment",
        "line_items[0][price_data][currency]": "usd",
        "line_items[0][price_data][unit_amount]": str(PRICE_CENTS),
        "line_items[0][price_data][product_data][name]": "Give It To Bonnie — the full video",
        "line_items[0][quantity]": "1",
        "allow_promotion_codes": "true",            # show the "Add promotion code" field
        "success_url": f"{base_url}/?paid=1&jid={jid}&sid={{CHECKOUT_SESSION_ID}}",
        "cancel_url": f"{base_url}/?canceled=1",
        "metadata[jid]": jid,
    }
    r = requests.post(f"{API}/checkout/sessions", auth=(_key(), ""), data=data, timeout=20)
    r.raise_for_status()
    j = r.json()
    return j["url"], j["id"]


def is_paid(sid):
    """True if the Checkout Session completed payment."""
    r = requests.get(f"{API}/checkout/sessions/{sid}", auth=(_key(), ""), timeout=20)
    r.raise_for_status()
    return r.json().get("payment_status") == "paid"
