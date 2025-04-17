import logging
from django.core.mail import EmailMultiAlternatives
from django.conf import settings
from google.cloud import firestore

logger = logging.getLogger(__name__)

def send_meeting_invites(meeting_title, meeting_date, start_time, end_time, agendas, participant_emails, organization):
    """
    Sends email invitations to meeting participants.
    """

    if not participant_emails:
        logger.error("No participant emails provided.")
        return {'error': 'No participants provided'}

    db = firestore.Client()

    # Find the admin user for this organization
    users_ref = db.collection('users')
    admin_query = users_ref.where('organization', '==', organization).where('role', '==', 'Admin').limit(1).get()

    admin_docs = list(admin_query)
    if not admin_docs:
        logger.error("Admin not found for the organization.")
        return {'error': 'Admin not found for the organization'}

    sender_email = settings.EMAIL_HOST_USER
    email_subject = f"Meeting Invitation: {meeting_title}"

    # Email content
    email_body_plain = f"""
    Dear Participant,

    You are invited to attend the upcoming meeting with the following details:

    Meeting Title: {meeting_title}
    Date: {meeting_date}
    Time: {start_time} - {end_time}
    Agenda: {', '.join(agendas)}

    Please note:
    - The meeting will be recorded.
    - Speak clearly for accurate discussions.
    - This is an English-only meeting.

    This is an automated message. Please do not reply.

    Best regards,
    {organization}
    """

    email_body_html = f"""
    <html>
    <body>
        <p><strong>Dear Participant,</strong></p>
        <p>You are invited to attend the upcoming meeting with the following details:</p>
        <p><strong>Meeting Title:</strong> {meeting_title}<br>
        <strong>Date:</strong> {meeting_date}<br>
        <strong>Time:</strong> {start_time} - {end_time}<br>
        <strong>Agenda:</strong> {', '.join(agendas)}</p>

        <p><strong>Important Notes:</strong></p>
        <ul>
            <li>The meeting will be recorded.</li>
            <li>Please speak clearly for accurate discussions.</li>
            <li>This is an <strong>English-only</strong> meeting.</li>
        </ul>

        <p><em>This is an automated message. Please do not reply.</em></p>
        <p>Best regards,<br><strong>{organization}</strong></p>
    </body>
    </html>
    """

    for participant_email in participant_emails:
        try:
            email_message = EmailMultiAlternatives(
                subject=email_subject,
                body=email_body_plain,
                from_email=sender_email,
                to=[participant_email]
            )
            email_message.attach_alternative(email_body_html, "text/html")
            email_message.send()
            
            logger.info(f"Email sent successfully to {participant_email}")

        except Exception as e:
            logger.error(f"Failed to send email to {participant_email}: {e}")

    return {'status': 'Emails sent successfully'}