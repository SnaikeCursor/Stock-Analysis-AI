"""Email notification service for trading signal alerts.

Uses Python's built-in ``smtplib`` — no extra dependencies.
Silently skips when SMTP credentials are not configured.

For Gmail: generate an App Password at
https://myaccount.google.com/apppasswords and use it as SMTP_PASSWORD.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(
        config.NOTIFY_EMAIL_TO
        and config.SMTP_HOST
        and config.SMTP_USER
        and config.SMTP_PASSWORD
    )


def send_email(subject: str, body_text: str, body_html: str | None = None) -> bool:
    """Send an email to the configured recipient.

    Returns True on success, False on failure or if not configured.
    """
    if not is_configured():
        logger.debug("Email notification skipped — not configured")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.SMTP_USER
    msg["To"] = config.NOTIFY_EMAIL_TO

    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.sendmail(config.SMTP_USER, config.NOTIFY_EMAIL_TO, msg.as_string())
        logger.info("Email notification sent to %s", config.NOTIFY_EMAIL_TO)
        return True
    except Exception:
        logger.exception("Failed to send email notification")
        return False


def notify_new_signal(
    signal_id: int,
    cutoff_date: str,
    n_positions: int,
    regime: str,
    top_picks: list[str] | None = None,
) -> bool:
    """Format and send a signal notification email."""
    picks = top_picks[:5] if top_picks else []
    picks_str = ", ".join(picks) if picks else "—"

    subject = f"Neues Trading-Signal #{signal_id} ({cutoff_date})"

    body_text = (
        f"Neues Trading-Signal #{signal_id}\n"
        f"{'=' * 40}\n\n"
        f"Cutoff-Datum:  {cutoff_date}\n"
        f"Regime:        {regime}\n"
        f"Positionen:    {n_positions}\n"
        f"Top-Picks:     {picks_str}\n\n"
        f"Jetzt im Dashboard pruefen und Rebalancing starten.\n"
    )

    rows_html = "".join(
        f"<tr><td style='padding:4px 12px;font-family:monospace'>{t}</td></tr>"
        for t in picks
    )

    body_html = f"""\
<div style="font-family:sans-serif;max-width:480px;margin:0 auto">
  <h2 style="color:#1a1a1a">Neues Trading-Signal #{signal_id}</h2>
  <table style="border-collapse:collapse;width:100%;margin:16px 0">
    <tr>
      <td style="padding:6px 0;color:#666">Cutoff-Datum</td>
      <td style="padding:6px 0;font-weight:600">{cutoff_date}</td>
    </tr>
    <tr>
      <td style="padding:6px 0;color:#666">Regime</td>
      <td style="padding:6px 0;font-weight:600">{regime}</td>
    </tr>
    <tr>
      <td style="padding:6px 0;color:#666">Positionen</td>
      <td style="padding:6px 0;font-weight:600">{n_positions}</td>
    </tr>
  </table>
  <h3 style="color:#1a1a1a;margin-top:20px">Top-Picks</h3>
  <table style="border-collapse:collapse;width:100%;background:#f8f8f8;border-radius:6px">
    {rows_html}
  </table>
  <p style="margin-top:24px;color:#666;font-size:14px">
    Jetzt im Dashboard pr&uuml;fen und Rebalancing starten.
  </p>
</div>
"""

    return send_email(subject, body_text, body_html)
