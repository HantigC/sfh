"""Every UnitOfWork implementation must pass the tests in this folder.

Adding a storage backend (e.g. Postgres) means adding one more param below.
"""

from collections.abc import Iterator

import pytest

from sfh.domain.ports.unit_of_work import UnitOfWorkFactory
from sfh.infrastructure.sqlite.unit_of_work import SqliteUnitOfWork, connect
from tests.fakes import FakeUnitOfWorkFactory


@pytest.fixture(params=["sqlite", "fake"])
def uow_factory(request: pytest.FixtureRequest) -> Iterator[UnitOfWorkFactory]:
    if request.param == "sqlite":
        conn = connect(":memory:")
        yield lambda: SqliteUnitOfWork(conn)
        conn.close()
    else:
        yield FakeUnitOfWorkFactory()
