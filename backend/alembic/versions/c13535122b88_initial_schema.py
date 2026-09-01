"""initial_schema

Revision ID: c13535122b88
Revises: 
Create Date: 2026-05-21 16:22:44.746533

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c13535122b88'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create the base schema (all tables except attribution_results,
    which is added by the next revision). checkfirst=True keeps this
    idempotent for databases already initialized via create_all."""
    from app.core.models import Base

    bind = op.get_bind()
    tables = [
        table for name, table in Base.metadata.tables.items()
        if name != "attribution_results"
    ]
    Base.metadata.create_all(bind=bind, tables=tables)


def downgrade() -> None:
    """Drop the base schema."""
    from app.core.models import Base

    bind = op.get_bind()
    tables = [
        table for name, table in Base.metadata.tables.items()
        if name != "attribution_results"
    ]
    Base.metadata.drop_all(bind=bind, tables=tables)
