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


def downgrade() -> None:
    op.drop_column("memory", "checked_character_rev")
