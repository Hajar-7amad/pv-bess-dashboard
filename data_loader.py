"""
CSV reading, timestamp detection and flexible column mapping.

Different EMS exports name the same quantity differently ("SOC", "State of Charge",
"Battery SOC (%)"...). This module

1. reads CSV files robustly (encoding, delimiter, decimal comma, junk header lines),
2. detects timestamp columns (single datetime column or separate date + time columns),
3. proposes a mapping from raw column names to the standardized schema in ``config``,
4. applies a user-confirmed mapping (with optional unit scale factors) and reports
   exactly how many values could not be parsed.

Nothing here invents data: unmapped variables simply stay absent.
"""
from __future__ import annotations

import csv
import difflib
import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import TIMESTAMP_SYNONYMS, VARIABLES

IGNORE = "— ignore —"
TS_SINGLE = "single"
TS_SPLIT = "date+time"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _sniff(text: str) -> Tuple[str, str]:
    """Return (delimiter, decimal) guessed from a text sample."""
    sample = "\n".join(text.splitlines()[:50])
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        delimiter = max(counts, key=counts.get)
    decimal = "."
    if delimiter == ";" and re.search(r"\d+,\d+", sample):
        decimal = ","
    return delimiter, decimal


def _find_header_row(lines: List[str], delimiter: str) -> int:
    """Skip leading metadata lines that EMS exports sometimes prepend."""
    if not lines:
        return 0
    counts = [ln.count(delimiter) for ln in lines[:30]]
    target = max(set(counts), key=counts.count) if counts else 0
    for i, c in enumerate(counts):
        if c == target and c > 0:
            return i
    return 0


def read_csv_flexible(buffer: io.BytesIO) -> Tuple[pd.DataFrame, Dict]:
    """Read a CSV of unknown dialect. Returns (dataframe, metadata)."""
    raw = buffer.read()
    text, encoding = None, None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and "\x00" in text:
                continue
            encoding = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        raise ValueError("Could not decode file (tried UTF-8, UTF-16, CP1252, Latin-1).")

    delimiter, decimal = _sniff(text)
    lines = text.splitlines()
    header_row = _find_header_row(lines, delimiter)
    df = pd.read_csv(
        io.StringIO(text), sep=delimiter, decimal=decimal, skiprows=header_row,
        skipinitialspace=True, low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")]
                or df.columns]
    df = df.dropna(how="all")
    meta = dict(encoding=encoding, delimiter=delimiter, decimal=decimal,
                header_row=header_row, n_rows=len(df), n_cols=df.shape[1])
    return df, meta


# ---------------------------------------------------------------------------
# Timestamp detection
# ---------------------------------------------------------------------------
def _parse_ratio(s: pd.Series, dayfirst: bool = False, n: int = 300) -> float:
    sample = s.dropna().astype(str).head(n)
    if sample.empty:
        return 0.0
    if pd.api.types.is_numeric_dtype(s):
        return 0.0  # pure numbers are almost never human timestamps in EMS exports
    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst, format="mixed")
    return float(parsed.notna().mean())


def guess_dayfirst(s: pd.Series) -> bool:
    """Decide day-first vs month-first for d/m/y-like strings.

    1. A leading field > 12 anywhere in the column proves day-first (second field > 12
       proves month-first).
    2. Otherwise both interpretations are parsed and the one giving fewer unparseable
       values and fewer backwards time steps (in file order) wins.
    """
    txt = s.dropna().astype(str)
    m = txt.str.extract(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
    if m.dropna().empty:
        return False
    a = pd.to_numeric(m[0], errors="coerce")
    b = pd.to_numeric(m[1], errors="coerce")
    if (a > 12).any():
        return True
    if (b > 12).any():
        return False
    scores = {}
    for df_ in (True, False):
        p = pd.to_datetime(txt, errors="coerce", dayfirst=df_, format="mixed")
        back = int((p.diff().dt.total_seconds() < 0).sum())
        scores[df_] = (int(p.isna().sum()), back)
    return scores[True] < scores[False]


@dataclass
class TimestampSpec:
    mode: str = TS_SINGLE          # single | date+time
    column: Optional[str] = None
    date_column: Optional[str] = None
    time_column: Optional[str] = None
    dayfirst: bool = False
    fmt: Optional[str] = None      # optional explicit strftime format
    tz_offset_h: float = 0.0       # shift applied after parsing (e.g. UTC -> local)

    def used_columns(self) -> List[str]:
        if self.mode == TS_SINGLE:
            return [self.column] if self.column else []
        return [c for c in (self.date_column, self.time_column) if c]


def detect_timestamp(df: pd.DataFrame) -> TimestampSpec:
    """Best-effort detection of the timestamp column(s)."""
    cols = list(df.columns)
    norm = {c: _normalize(c) for c in cols}

    # separate date + time columns?
    date_cols = [c for c in cols if norm[c] in ("date", "day", "log date")]
    time_cols = [c for c in cols if norm[c] in ("time", "hour", "log time", "clock")]
    if date_cols and time_cols:
        combo = df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str)
        if _parse_ratio(combo) > 0.8:
            return TimestampSpec(mode=TS_SPLIT, date_column=date_cols[0],
                                 time_column=time_cols[0], dayfirst=guess_dayfirst(df[date_cols[0]]))

    scored = []
    for c in cols:
        name_score = max((1.0 if norm[c] == syn else 0.6 if syn in norm[c] else 0.0)
                         for syn in TIMESTAMP_SYNONYMS)
        if name_score == 0 and not df[c].dtype == object:
            continue
        ratio = _parse_ratio(df[c])
        if ratio > 0.8:
            scored.append((name_score + ratio, c))
    if scored:
        best = max(scored)[1]
        return TimestampSpec(mode=TS_SINGLE, column=best, dayfirst=guess_dayfirst(df[best]))
    return TimestampSpec(mode=TS_SINGLE, column=None)


def build_timestamp(df: pd.DataFrame, spec: TimestampSpec) -> pd.Series:
    """Parse the timestamp according to ``spec``. Unparseable entries become NaT."""
    if spec.mode == TS_SPLIT:
        if not (spec.date_column and spec.time_column):
            raise ValueError("Select both a date and a time column.")
        s = df[spec.date_column].astype(str).str.strip() + " " + df[spec.time_column].astype(str).str.strip()
    else:
        if not spec.column:
            raise ValueError("No timestamp column selected.")
        s = df[spec.column]
    if pd.api.types.is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, errors="coerce")
    elif spec.fmt:
        ts = pd.to_datetime(s.astype(str).str.strip(), format=spec.fmt, errors="coerce")
    else:
        ts = pd.to_datetime(s.astype(str).str.strip(), errors="coerce",
                            dayfirst=spec.dayfirst, format="mixed")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    if spec.tz_offset_h:
        ts = ts + pd.Timedelta(hours=spec.tz_offset_h)
    return ts


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------
_UNIT_RE = re.compile(r"[\(\[\{].*?[\)\]\}]")


def _normalize(name: str) -> str:
    s = str(name).lower()
    s = _UNIT_RE.sub(" ", s)
    s = re.sub(r"[_\-./:%#]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _unit_hint(name: str) -> Optional[str]:
    low = str(name).lower()
    if "kwh" in low or "mwh" in low:
        return "energy"
    if re.search(r"\bkw\b|\(kw\)|\[kw\]|\bw\b|\(w\)", low) and "w/m" not in low:
        return "power"
    if "w/m" in low or "w m2" in low:
        return "irradiance"
    if "%" in low:
        return "percent"
    if "°c" in low or "degc" in low or "(c)" in low:
        return "temperature"
    return None


_HINT_GROUPS = {
    "energy": {"pv_energy", "total_energy"},
    "power": {"pv_power", "load_power", "grid_power", "battery_charge_power",
              "battery_discharge_power", "battery_power"},
    "irradiance": {"solar_irradiance"},
    "percent": {"battery_soc"},
    "temperature": {"ambient_temperature", "pv_cell_temperature", "inverter_temperature"},
}


def score_column(name: str, var_key: str) -> float:
    """Similarity in [0, 1+] between a raw column name and a standardized variable."""
    var = VARIABLES[var_key]
    norm = _normalize(name)
    if not norm:
        return 0.0
    best = 0.0
    for syn in var.synonyms + [_normalize(var.label), var.key.replace("_", " ")]:
        if norm == syn:
            best = max(best, 1.0)
        elif re.search(rf"\b{re.escape(syn)}\b", norm):
            # longer synonym covering more of the name = stronger evidence
            best = max(best, 0.45 + 0.5 * len(syn) / max(len(norm), 1))
        else:
            best = max(best, 0.6 * difflib.SequenceMatcher(None, norm, syn).ratio())
    hint = _unit_hint(name)
    if hint:
        best += 0.15 if var_key in _HINT_GROUPS.get(hint, set()) else -0.2
    return best


def suggest_mapping(columns: List[str], exclude: Optional[List[str]] = None,
                    min_score: float = 0.55) -> Dict[str, Tuple[str, float]]:
    """Propose {raw_column: (variable_key, score)}; each variable used at most once."""
    exclude = set(exclude or [])
    candidates = []
    for c in columns:
        if c in exclude:
            continue
        for k in VARIABLES:
            s = score_column(c, k)
            if s >= min_score:
                candidates.append((s, c, k))
    candidates.sort(reverse=True)
    used_cols, used_vars, out = set(), set(), {}
    for s, c, k in candidates:
        if c in used_cols or k in used_vars:
            continue
        out[c] = (k, round(s, 2))
        used_cols.add(c)
        used_vars.add(k)
    return out


@dataclass
class FileMapping:
    """User-confirmed mapping for one raw file."""

    file_name: str
    ts: TimestampSpec
    columns: Dict[str, str] = field(default_factory=dict)   # raw col -> var key
    scale: Dict[str, float] = field(default_factory=dict)   # raw col -> multiplier


@dataclass
class StandardizeReport:
    file_name: str
    rows_in: int
    rows_bad_timestamp: int
    rows_out: int
    invalid_numeric: Dict[str, int]
    mapped: Dict[str, str]
    first: Optional[pd.Timestamp]
    last: Optional[pd.Timestamp]
    median_interval_s: Optional[float]


def standardize(df: pd.DataFrame, mapping: FileMapping) -> Tuple[pd.DataFrame, StandardizeReport]:
    """Apply a mapping: parse timestamp, rename, coerce numerics, apply scale factors.

    Rows whose timestamp cannot be parsed are counted and excluded (they cannot be placed
    in time); non-numeric values become NaN and are counted per variable.
    """
    ts = build_timestamp(df, mapping.ts)
    bad_ts = int(ts.isna().sum())
    out = pd.DataFrame({"timestamp": ts})
    invalid = {}
    for raw, key in mapping.columns.items():
        if key in (None, "", IGNORE) or raw not in df.columns:
            continue
        col = df[raw]
        if col.dtype == object:
            cleaned = (col.astype(str).str.strip()
                       .str.replace(r"[^\d,.\-eE+]", "", regex=True))
            # decimal comma that survived reading (e.g. "12,5")
            cleaned = cleaned.where(~cleaned.str.fullmatch(r"-?\d+,\d+"),
                                    cleaned.str.replace(",", ".", regex=False))
            num = pd.to_numeric(cleaned, errors="coerce")
        else:
            num = pd.to_numeric(col, errors="coerce")
        invalid[key] = int((num.isna() & col.notna() & (col.astype(str).str.strip() != "")).sum())
        num = num * float(mapping.scale.get(raw, 1.0) or 1.0)
        if key in out.columns:  # two raw columns mapped to one variable: combine
            out[key] = out[key].combine_first(num)
        else:
            out[key] = num
    out = out[out["timestamp"].notna()].reset_index(drop=True)
    dt = out["timestamp"].sort_values().diff().dt.total_seconds()
    rep = StandardizeReport(
        file_name=mapping.file_name, rows_in=len(df), rows_bad_timestamp=bad_ts,
        rows_out=len(out), invalid_numeric=invalid,
        mapped={k: v for k, v in mapping.columns.items() if v not in (None, "", IGNORE)},
        first=out["timestamp"].min() if len(out) else None,
        last=out["timestamp"].max() if len(out) else None,
        median_interval_s=float(dt[dt > 0].median()) if (dt > 0).any() else None,
    )
    return out, rep


def auto_mapping(file_name: str, df: pd.DataFrame) -> FileMapping:
    """Fully automatic mapping (used for demo data and as UI starting point)."""
    ts = detect_timestamp(df)
    sugg = suggest_mapping(list(df.columns), exclude=ts.used_columns())
    return FileMapping(file_name=file_name, ts=ts,
                       columns={c: k for c, (k, _) in sugg.items()},
                       scale={c: 1.0 for c in sugg})


def coverage_table(std_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Which standardized variables each file provides (True/False)."""
    rows = []
    for name, f in std_frames.items():
        row = {"file": name}
        for k in VARIABLES:
            row[k] = k in f.columns and f[k].notna().any()
        rows.append(row)
    return pd.DataFrame(rows).set_index("file") if rows else pd.DataFrame()
