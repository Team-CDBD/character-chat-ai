"""Remove applied event bodies, retaining receipts and deduplication keys.

FAILED bodies are kept: they are needed to inspect and re-queue the event.
"""

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute("UPDATE event_inbox SET payload = '{}'::jsonb WHERE status = 'APPLIED'")


def downgrade() -> None:
    # Processed bodies cannot be reconstructed and are never used for replay.
    pass
