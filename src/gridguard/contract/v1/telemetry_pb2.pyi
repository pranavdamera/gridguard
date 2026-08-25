import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QualityFlag(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUALITY_FLAG_OK: _ClassVar[QualityFlag]
    QUALITY_FLAG_OUT_OF_RANGE: _ClassVar[QualityFlag]
    QUALITY_FLAG_FROZEN: _ClassVar[QualityFlag]
    QUALITY_FLAG_STALE: _ClassVar[QualityFlag]
    QUALITY_FLAG_INTERPOLATED: _ClassVar[QualityFlag]
    QUALITY_FLAG_MISSING: _ClassVar[QualityFlag]
    QUALITY_FLAG_OUT_OF_ORDER: _ClassVar[QualityFlag]

class NodeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_STATE_UNSPECIFIED: _ClassVar[NodeState]
    NODE_STATE_ONLINE: _ClassVar[NodeState]
    NODE_STATE_OFFLINE_UNEXPECTED: _ClassVar[NodeState]
    NODE_STATE_OFFLINE_CLEAN: _ClassVar[NodeState]
    NODE_STATE_DEGRADED: _ClassVar[NodeState]
QUALITY_FLAG_OK: QualityFlag
QUALITY_FLAG_OUT_OF_RANGE: QualityFlag
QUALITY_FLAG_FROZEN: QualityFlag
QUALITY_FLAG_STALE: QualityFlag
QUALITY_FLAG_INTERPOLATED: QualityFlag
QUALITY_FLAG_MISSING: QualityFlag
QUALITY_FLAG_OUT_OF_ORDER: QualityFlag
NODE_STATE_UNSPECIFIED: NodeState
NODE_STATE_ONLINE: NodeState
NODE_STATE_OFFLINE_UNEXPECTED: NodeState
NODE_STATE_OFFLINE_CLEAN: NodeState
NODE_STATE_DEGRADED: NodeState

class TelemetryRecord(_message.Message):
    __slots__ = ("asset_id", "site_id", "channel", "unit", "value", "event_time", "ingest_time", "sequence", "quality")
    ASSET_ID_FIELD_NUMBER: _ClassVar[int]
    SITE_ID_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    EVENT_TIME_FIELD_NUMBER: _ClassVar[int]
    INGEST_TIME_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    asset_id: str
    site_id: str
    channel: str
    unit: str
    value: float
    event_time: _timestamp_pb2.Timestamp
    ingest_time: _timestamp_pb2.Timestamp
    sequence: int
    quality: int
    def __init__(self, asset_id: _Optional[str] = ..., site_id: _Optional[str] = ..., channel: _Optional[str] = ..., unit: _Optional[str] = ..., value: _Optional[float] = ..., event_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., ingest_time: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., sequence: _Optional[int] = ..., quality: _Optional[int] = ...) -> None: ...

class TelemetryBatch(_message.Message):
    __slots__ = ("schema_version", "agent_id", "site_id", "records", "sent_at", "is_replay", "buffered_remaining")
    SCHEMA_VERSION_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    SITE_ID_FIELD_NUMBER: _ClassVar[int]
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    SENT_AT_FIELD_NUMBER: _ClassVar[int]
    IS_REPLAY_FIELD_NUMBER: _ClassVar[int]
    BUFFERED_REMAINING_FIELD_NUMBER: _ClassVar[int]
    schema_version: str
    agent_id: str
    site_id: str
    records: _containers.RepeatedCompositeFieldContainer[TelemetryRecord]
    sent_at: _timestamp_pb2.Timestamp
    is_replay: bool
    buffered_remaining: int
    def __init__(self, schema_version: _Optional[str] = ..., agent_id: _Optional[str] = ..., site_id: _Optional[str] = ..., records: _Optional[_Iterable[_Union[TelemetryRecord, _Mapping]]] = ..., sent_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., is_replay: _Optional[bool] = ..., buffered_remaining: _Optional[int] = ...) -> None: ...

class NodeStatus(_message.Message):
    __slots__ = ("agent_id", "site_id", "state", "observed_at", "detail", "buffered_records", "last_delivered_sequence")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    SITE_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_AT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    BUFFERED_RECORDS_FIELD_NUMBER: _ClassVar[int]
    LAST_DELIVERED_SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    site_id: str
    state: NodeState
    observed_at: _timestamp_pb2.Timestamp
    detail: str
    buffered_records: int
    last_delivered_sequence: int
    def __init__(self, agent_id: _Optional[str] = ..., site_id: _Optional[str] = ..., state: _Optional[_Union[NodeState, str]] = ..., observed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., detail: _Optional[str] = ..., buffered_records: _Optional[int] = ..., last_delivered_sequence: _Optional[int] = ...) -> None: ...
