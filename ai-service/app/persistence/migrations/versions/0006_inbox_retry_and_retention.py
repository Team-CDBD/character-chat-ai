"""Retry schedule for failed lifecycle events, and indexes for the 24h retention sweep.

* ``event_inbox.attempts`` / ``next_attempt_at``: a FAILED event is retried with backoff instead of
  blocking its aggregate forever; re-sending the same eventId re-arms it immediately.
* ``ix_snapshot_expires``: the sweep deletes expired context snapshots by ``expires_at``.
* ``ix_job_retention``: finished generations whose request body / stream events are still stored,
  ordered by ``completed_at``; once scrubbed (payload ``{}``) a row leaves the index.
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    op.add_column("event_inbox", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("event_inbox", sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True))
    # Events that failed before this migration get one immediate retry.
    op.execute("UPDATE event_inbox SET attempts = 1, next_attempt_at = now() WHERE status = 'FAILED'")
    op.create_index("ix_snapshot_expires", "context_snapshot", ["expires_at"], unique=False)
    op.create_index(
        "ix_job_retention", "ai_job", ["completed_at"], unique=False,
        postgresql_where=sa.text("kind = 'GENERATION' AND completed_at IS NOT NULL AND payload <> '{}'::jsonb"),
    )


def downgrade() -> None:
    op.drop_index("ix_job_retention", table_name="ai_job")
    op.drop_index("ix_snapshot_expires", table_name="context_snapshot")
    op.drop_column("event_inbox", "next_attempt_at")
    op.drop_column("event_inbox", "attempts")
