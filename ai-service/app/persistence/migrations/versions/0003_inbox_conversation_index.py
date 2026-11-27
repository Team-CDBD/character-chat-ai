"""Index inbox rows by conversation so reset/delete scrubbing does not scan the whole inbox."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.add_column("event_inbox", sa.Column("conversation_id", sa.String(length=64), nullable=True))
    op.execute("UPDATE event_inbox SET conversation_id = payload->>'conversationId' WHERE payload->>'conversationId' IS NOT NULL")
    op.create_index("ix_inbox_conversation", "event_inbox", ["tenant_id", "conversation_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_inbox_conversation", table_name="event_inbox")
    op.drop_column("event_inbox", "conversation_id")
