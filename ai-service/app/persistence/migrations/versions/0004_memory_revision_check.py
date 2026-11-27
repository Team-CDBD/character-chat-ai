"""Track the latest character revision each memory was checked against.

Before: a character/shared memory was only usable under the exact revision it was said in, so any
character edit hid all of them. Now a revision-check job compares them with the new card and hides
only the ones that conflict (``conflicts_with_revision``); the rest get ``checked_character_rev``
bumped. Existing rows start as checked against the revision of their source turn.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"


def upgrade() -> None:
    op.add_column("memory", sa.Column("checked_character_rev", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE memory m SET checked_character_rev = t.character_rev
        FROM committed_turn t
        WHERE t.tenant_id = m.tenant_id AND t.conversation_id = m.conversation_id AND t.epoch = m.epoch
          AND t.turn_seq = m.source_turn_seq
        """
    )
    # Conversations edited before this migration still hide those older character facts: queue one check
    # per active conversation so they get the new behaviour without waiting for the next edit.
    op.execute(
        """
        INSERT INTO ai_job (id, kind, tenant_id, owner_id, conversation_id, epoch, turn_seq, attempt_id,
            payload_hash, payload, status, fencing_token, run_count, next_event_seq, degraded, created_at, updated_at)
        SELECT gen_random_uuid()::text, 'REVISION', c.tenant_id, c.owner_id, c.conversation_id, c.epoch, 0,
               'revision:' || md5(c.conversation_id || ':' || c.epoch || ':' || c.character_rev) || ':' || 'migrate',
               '', jsonb_build_object('characterRevision', c.character_rev), 'ACCEPTED', 0, 0, 1, '[]'::jsonb,
               now(), now()
        FROM ai_conversation c
        WHERE c.status = 'ACTIVE' AND EXISTS (
            SELECT 1 FROM memory m
            WHERE m.tenant_id = c.tenant_id AND m.conversation_id = c.conversation_id AND m.epoch = c.epoch
              AND m.valid_to IS NULL AND m.subject <> 'user'
              AND COALESCE(m.checked_character_rev, 0) < c.character_rev)
        ON CONFLICT (kind, attempt_id) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("DELETE FROM ai_job WHERE kind = 'REVISION'")
    op.drop_column("memory", "checked_character_rev")
