"""Remove processed event bodies, retaining receipts and deduplication keys."""

from alembic import op

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.execute("UPDATE event_inbox SET payload = '{}'::jsonb WHERE status IN ('APPLIED', 'FAILED')")


def downgrade() -> None:
    # Processed bodies cannot be reconstructed and are never used for replay.
    pass
