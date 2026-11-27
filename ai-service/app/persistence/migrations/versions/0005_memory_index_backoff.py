"""Per-row retry schedule for embedding failures.

A memory whose embedding keeps failing is retried later and later (``index_retry_at``) instead of
being picked first every time, so it cannot hold back other rows that could be re-indexed.
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.add_column("memory", sa.Column("index_attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("memory", sa.Column("index_retry_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("memory", "index_retry_at")
    op.drop_column("memory", "index_attempts")
