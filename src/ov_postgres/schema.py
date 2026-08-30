"""Parsing of OpenViking collection schemas into typed field specs.

OpenViking describes a collection as a ``Fields`` list of
``{"FieldName": ..., "FieldType": ..., ...}`` dicts plus a ``ScalarIndex``
list naming the fields that must be filterable. This module turns that
description into :class:`FieldSpec` objects.

Every declared field becomes a real, typed column so filters compile to
ordinary SQL predicates over ordinary indexes. Anything written that is *not*
in the schema is preserved in a JSONB ``extra`` column rather than dropped.

The SQL column type for each field lives in :mod:`ov_postgres.ddl`, so the
mapping has one home; this module only decides which field types are
representable at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

# Field types OpenViking can declare that have a scalar column form. Mirrors
# the non-None entries of ``DataProcessor.ENGINE_SCALAR_TYPE_MAP`` in
# openviking/storage/vectordb/utils/data_processor.py. ``date_time`` is stored
# as epoch milliseconds in a BIGINT, exactly as the native engine does, so
# range filters on timestamps compare integers on both backends.
SCALAR_TYPES: frozenset[str] = frozenset(
    {
        "string",
        "text",
        "path",
        "int64",
        "float32",
        "bool",
        "list<string>",
        "list<int64>",
        "date_time",
        "sparse_vector",
    }
)

# Binary media fields have no scalar form; they round-trip via ``extra``.
UNSUPPORTED_TYPES: frozenset[str] = frozenset({"image", "video"})

ARRAY_TYPES: frozenset[str] = frozenset({"list<string>", "list<int64>"})

# Field types whose values are textual and can be searched as text.
TEXT_TYPES: frozenset[str] = frozenset({"string", "text", "path"})

GEO_LON_SUFFIX = "_lon"
GEO_LAT_SUFFIX = "_lat"


def normalize_field_type(field_type: object) -> str:
    """Return a field type as a plain string.

    OpenViking passes either a string or an enum member, depending on how the
    schema was built.

    Parameters
    ----------
    field_type :
        A field type string or an enum with a ``value`` attribute.

    Returns
    -------
    str
        The field type name.
    """
    value = getattr(field_type, "value", None)
    if value is not None:
        return str(value)
    return str(field_type)


@dataclass(frozen=True)
class FieldSpec:
    """One column of a collection table.

    Attributes
    ----------
    name : str
        Column name, taken verbatim from ``FieldName``.
    ov_type : str
        OpenViking field type, such as ``"path"`` or ``"list<string>"``.
    is_primary : bool
        Whether this column is the collection's primary key.
    dim : int | None
        Declared dimension, set only for ``vector`` fields.
    declared_default : object
        The schema's own ``DefaultValue``, which the engine prefers over the
        per-type default. ``None`` when the schema declares none.
    """

    name: str
    ov_type: str
    is_primary: bool = False
    dim: int | None = None
    declared_default: object = None

    @property
    def is_vector(self) -> bool:
        """Whether the column holds a dense vector."""
        return self.ov_type == "vector"

    @property
    def is_sparse(self) -> bool:
        """Whether the column holds a sparse vector."""
        return self.ov_type == "sparse_vector"

    @property
    def is_path(self) -> bool:
        """Whether the column holds a URI path subject to depth scoping."""
        return self.ov_type == "path"

    @property
    def is_datetime(self) -> bool:
        """Whether the column holds a timestamp stored as epoch milliseconds."""
        return self.ov_type == "date_time"

    @property
    def is_array(self) -> bool:
        """Whether the column holds a PostgreSQL array."""
        return self.ov_type in ARRAY_TYPES

    @property
    def is_geo(self) -> bool:
        """Whether the column expands into a longitude/latitude pair."""
        return self.ov_type == "geo_point"

    @property
    def is_textual(self) -> bool:
        """Whether the column's values can be searched as text."""
        return self.ov_type in TEXT_TYPES or self.ov_type == "list<string>"

    @property
    def selectable(self) -> bool:
        """Whether the column is returned by default in search output."""
        return not (self.is_vector or self.is_sparse)


@dataclass
class CollectionSchema:
    """Parsed form of an OpenViking collection meta dict.

    Attributes
    ----------
    name : str
        Collection name.
    description : str
        Human-readable description carried in the registry.
    fields : list[FieldSpec]
        One entry per representable declared field.
    scalar_index : list[str]
        Field names that must be filterable, and so get an index.
    fulltext : list[dict[str, Any]]
        The schema's ``FullText`` section, retained but not interpreted.
    raw : dict[str, Any]
        The original meta dict, returned by ``get_meta_data``.
    """

    name: str
    description: str = ""
    fields: list[FieldSpec] = dc_field(default_factory=list)
    scalar_index: list[str] = dc_field(default_factory=list)
    fulltext: list[dict[str, Any]] = dc_field(default_factory=list)
    raw: dict[str, Any] = dc_field(default_factory=dict)

    def by_name(self, name: str) -> FieldSpec | None:
        """Return the spec for a field, or ``None`` if undeclared.

        Parameters
        ----------
        name :
            Field name to look up.

        Returns
        -------
        FieldSpec | None
            The matching spec, or ``None``.
        """
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None

    @property
    def primary_key(self) -> FieldSpec:
        """The collection's primary-key field.

        Raises
        ------
        ValueError
            If no field is marked as the primary key.
        """
        for spec in self.fields:
            if spec.is_primary:
                return spec
        raise ValueError(f"Collection {self.name!r} declares no primary key")

    @property
    def vector_field(self) -> FieldSpec | None:
        """The dense-vector field, if the collection declares one."""
        for spec in self.fields:
            if spec.is_vector:
                return spec
        return None

    @property
    def sparse_field(self) -> FieldSpec | None:
        """The sparse-vector field, if the collection declares one."""
        for spec in self.fields:
            if spec.is_sparse:
                return spec
        return None

    @property
    def field_types(self) -> dict[str, str]:
        """Field name to OpenViking type, as ``matches_filter`` expects."""
        return {spec.name: spec.ov_type for spec in self.fields}

    def selectable_names(self) -> list[str]:
        """Return the columns included in search output by default.

        Returns
        -------
        list[str]
            Field names excluding dense and sparse vectors, which are large
            and rarely wanted.
        """
        return [spec.name for spec in self.fields if spec.selectable]

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> CollectionSchema:
        """Parse an OpenViking collection meta dict.

        Parameters
        ----------
        meta :
            The collection schema, as built by ``CollectionSchemas``.

        Returns
        -------
        CollectionSchema
            The parsed schema.

        Raises
        ------
        ValueError
            If the meta has no collection name, declares no primary key, or
            contains a field type this backend cannot represent.
        """
        name = meta.get("CollectionName") or meta.get("name")
        if not name:
            raise ValueError("Collection meta is missing 'CollectionName'")

        specs: list[FieldSpec] = []
        for raw_field in meta.get("Fields", []) or []:
            spec = _field_spec(raw_field)
            if spec is not None:
                specs.append(spec)

        if not any(spec.is_primary for spec in specs):
            raise ValueError(f"Collection {name!r} declares no primary key field")

        return cls(
            name=str(name),
            description=str(meta.get("Description") or ""),
            fields=specs,
            scalar_index=[str(f) for f in (meta.get("ScalarIndex") or [])],
            fulltext=list(meta.get("FullText") or []),
            raw=dict(meta),
        )


def _field_spec(raw_field: dict[str, Any]) -> FieldSpec | None:
    """Build a spec for one declared field, or ``None`` to skip it.

    Parameters
    ----------
    raw_field :
        One entry of the schema's ``Fields`` list.

    Returns
    -------
    FieldSpec | None
        The spec, or ``None`` for a field with no column representation.

    Raises
    ------
    ValueError
        If a vector field omits its dimension, or the type is unknown.
    """
    name = raw_field.get("FieldName")
    if not name:
        return None
    ov_type = normalize_field_type(raw_field.get("FieldType"))

    if ov_type in UNSUPPORTED_TYPES:
        return None

    if ov_type == "vector":
        dim = raw_field.get("Dim") or raw_field.get("Dimension")
        if not dim:
            raise ValueError(f"Vector field {name!r} is missing 'Dim'")
        return FieldSpec(name=str(name), ov_type="vector", is_primary=False, dim=int(dim))

    if ov_type == "geo_point":
        # Stored as a pair of real columns, matching the native engine's
        # `_lon`/`_lat` expansion.
        return FieldSpec(name=str(name), ov_type="geo_point")

    if ov_type not in SCALAR_TYPES:
        raise ValueError(f"Unsupported field type {ov_type!r} for field {name!r}")

    return FieldSpec(
        name=str(name),
        ov_type=ov_type,
        is_primary=bool(raw_field.get("IsPrimaryKey")),
        declared_default=raw_field.get("DefaultValue"),
    )


def fulltext_candidates(schema: CollectionSchema, configured: Iterable[str]) -> list[str]:
    """Return the text columns to include in the tsvector.

    Fields absent from the schema, or that are not textual, are skipped rather
    than raising: an upstream schema change should not break startup.

    Parameters
    ----------
    schema :
        The parsed collection schema.
    configured :
        Field names requested via the ``keyword_fields`` option.

    Returns
    -------
    list[str]
        Names of textual columns present in the schema, in the configured
        order.
    """
    chosen: list[str] = []
    for name in configured:
        spec = schema.by_name(name)
        if spec is not None and spec.is_textual:
            chosen.append(name)
    return chosen


# Values the native engine substitutes for an omitted field. Mirrors
# ``TYPE_DEFAULTS`` in openviking/storage/vectordb/utils/data_processor.py,
# which builds them into the pydantic validator every write passes through.
# Storing NULL instead would make an omitted `level` absent here and ``0``
# there, so a filter for ``level == 0`` would find the row on the built-in
# backend and miss it on this one.
TYPE_DEFAULTS: dict[str, Any] = {
    "int64": 0,
    "float32": 0.0,
    "string": "",
    "bool": False,
    "list<string>": [],
    "list<int64>": [],
    "text": "",
    "path": "",
    "date_time": "",
    "geo_point": "",
    "sparse_vector": {},
}


def default_for(spec: FieldSpec) -> object:
    """Return the value the native engine stores for an omitted field.

    Parameters
    ----------
    spec :
        The declared field.

    Returns
    -------
    object
        The engine's default, or ``None`` for a type it leaves unset.
    """
    if spec.is_primary or spec.is_vector:
        return None
    # `convert_fields_dict_for_index` drops date_time and geo_point when their
    # value is empty, and `LocalCollection._write_data_list` pops the vector and
    # sparse-vector keys outright, so the engine index never holds a default for
    # any of them.
    if spec.ov_type in ("date_time", "geo_point", "sparse_vector"):
        return None
    # The declared default is checked only after the exclusions, and only when
    # its type fits the column. Applying it first let a DefaultValue of ""
    # reinstate a NULL timestamp on every backfill pass, and a DefaultValue of
    # the wrong type reach a typed column and fail every write.
    declared = spec.declared_default
    if declared is not None and _fits(spec, declared):
        return declared
    value = TYPE_DEFAULTS.get(spec.ov_type)
    return list(value) if isinstance(value, list) else value


def _fits(spec: FieldSpec, value: object) -> bool:
    """Return whether a declared default is storable in the column.

    The engine does not validate ``DefaultValue`` -- pydantic skips defaults --
    so a mistyped one is silently accepted there and would fail every write
    here. Ignoring it and using the type default keeps the collection usable.

    Parameters
    ----------
    spec :
        The declared field.
    value :
        The schema's ``DefaultValue``.

    Returns
    -------
    bool
        True when the value matches the column's type.
    """
    if spec.ov_type in ("string", "text", "path"):
        return isinstance(value, str)
    if spec.ov_type == "bool":
        return isinstance(value, bool)
    if spec.ov_type == "int64":
        return isinstance(value, int) and not isinstance(value, bool)
    if spec.ov_type == "float32":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec.ov_type in ("list<string>", "list<int64>"):
        return isinstance(value, list)
    return False
