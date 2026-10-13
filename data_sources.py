"""
Data-source abstraction.

The dashboard never reads files directly: it asks a DataSource for raw tables. This lets
the same analysis pages run on historical CSV exports today and on a live EMS feed later.

Implemented:
    CSVDataSource        uploaded or on-disk CSV files (fully implemented)

Interface placeholders (clean contracts, not yet connected):
    DatabaseDataSource   SQL historian (e.g. PostgreSQL / TimescaleDB / MS-SQL)
    APIDataSource        REST/JSON endpoint exposed by the EMS vendor
    LiveEMSDataSource    streaming / polling connection (Modbus-TCP, OPC-UA, MQTT)

IMPORTANT: every source is READ-ONLY. This platform is for monitoring, analytics,
comparison, prediction and explainability. It never sends battery control commands.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from modules.data_loader import read_csv_flexible


@dataclass
class RawTable:
    """A raw table as delivered by a source, before column mapping."""

    name: str
    df: pd.DataFrame
    meta: Dict = field(default_factory=dict)


class DataSource(ABC):
    """Read-only contract shared by all data sources."""

    kind: str = "abstract"
    is_live: bool = False

    @abstractmethod
    def list_tables(self) -> List[str]:
        """Names of the tables/files/streams this source offers."""

    @abstractmethod
    def read_table(self, name: str) -> RawTable:
        """Return one raw table (unmapped column names)."""

    def read_all(self) -> List[RawTable]:
        return [self.read_table(n) for n in self.list_tables()]

    def latest(self, standardized: pd.DataFrame) -> Optional[pd.Series]:
        """Most recent standardized record. Historical sources return the last row."""
        if standardized is None or standardized.empty:
            return None
        return standardized.iloc[-1]

    def describe(self) -> str:
        return f"{self.kind} data source"


class CSVDataSource(DataSource):
    """CSV files supplied as uploaded byte buffers or file paths."""

    kind = "CSV"

    def __init__(self, files: Dict[str, Union[bytes, str, Path]]):
        self._files = dict(files)
        self._cache: Dict[str, RawTable] = {}

    def list_tables(self) -> List[str]:
        return list(self._files.keys())

    def read_table(self, name: str) -> RawTable:
        if name in self._cache:
            return self._cache[name]
        src = self._files[name]
        if isinstance(src, (str, Path)):
            data = Path(src).read_bytes()
        else:
            data = src
        df, meta = read_csv_flexible(io.BytesIO(data))
        meta["size_bytes"] = len(data)
        table = RawTable(name=name, df=df, meta=meta)
        self._cache[name] = table
        return table

    def describe(self) -> str:
        return f"CSV source with {len(self._files)} file(s)"


class DatabaseDataSource(DataSource):
    """Placeholder for an SQL historian connection.

    Expected configuration: SQLAlchemy URL, table names, timestamp column.
    Implementation sketch: ``pd.read_sql(query, engine, params={...})`` per table.
    """

    kind = "Database"

    def __init__(self, connection_url: str = "", tables: Optional[List[str]] = None):
        self.connection_url = connection_url
        self.tables = tables or []

    def list_tables(self) -> List[str]:
        return list(self.tables)

    def read_table(self, name: str) -> RawTable:
        raise NotImplementedError("DatabaseDataSource is an interface placeholder.")


class APIDataSource(DataSource):
    """Placeholder for a REST/JSON EMS API (e.g. EPower Monitor web service)."""

    kind = "API"

    def __init__(self, base_url: str = "", endpoints: Optional[Dict[str, str]] = None,
                 auth_token: Optional[str] = None):
        self.base_url = base_url
        self.endpoints = endpoints or {}
        self.auth_token = auth_token

    def list_tables(self) -> List[str]:
        return list(self.endpoints)

    def read_table(self, name: str) -> RawTable:
        raise NotImplementedError("APIDataSource is an interface placeholder.")


class LiveEMSDataSource(DataSource):
    """Placeholder for a live, read-only EMS connection.

    Intended transport: Modbus-TCP / OPC-UA / MQTT polling. ``latest()`` would return the
    newest standardized record; the Live Monitor page already polls ``latest()`` on a
    configurable refresh interval, so wiring a real transport here is the only change
    needed. Write/control functions are intentionally absent.
    """

    kind = "Live EMS"
    is_live = True

    def __init__(self, host: str = "", port: int = 0, protocol: str = "modbus-tcp",
                 tag_map: Optional[Dict[str, str]] = None):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.tag_map = tag_map or {}

    def list_tables(self) -> List[str]:
        return ["live"]

    def read_table(self, name: str) -> RawTable:
        raise NotImplementedError("LiveEMSDataSource is an interface placeholder.")

    def latest(self, standardized: pd.DataFrame = None) -> Optional[pd.Series]:
        raise NotImplementedError("LiveEMSDataSource is an interface placeholder.")


AVAILABLE_SOURCES = {
    "CSV files": CSVDataSource,
    "Database (planned)": DatabaseDataSource,
    "EMS API (planned)": APIDataSource,
    "Live EMS (planned)": LiveEMSDataSource,
}
