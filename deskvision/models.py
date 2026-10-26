"""S3 persistence schema: task metadata, independent routes, and file index."""

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserRecord(Base):
    __tablename__ = 'users'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class LoginSession(Base):
    __tablename__ = 'login_sessions'

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), nullable=False, index=True)
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class RunRecord(Base):
    __tablename__ = 'runs'
    __table_args__ = (
        CheckConstraint("status IN ('queued','running','succeeded','partial_failed','failed')",
                        name='ck_runs_status'),
        CheckConstraint("route IN ('yolo','deepseek','both')", name='ck_runs_route'),
        Index('ix_runs_created_at', 'created_at'),
        Index('ix_runs_status_created_at', 'status', 'created_at'),
    )

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str | None] = mapped_column(ForeignKey('users.id'), index=True)
    route: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    started_at: Mapped[str | None] = mapped_column(String(40))
    finished_at: Mapped[str | None] = mapped_column(String(40))
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    input_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    input_width: Mapped[int] = mapped_column(Integer, nullable=False)
    input_height: Mapped[int] = mapped_column(Integer, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    input_path: Mapped[str] = mapped_column(Text, nullable=False)
    run_dir: Mapped[str] = mapped_column(Text, nullable=False)
    comparison: Mapped[str | None] = mapped_column(Text)
    error: Mapped[dict | None] = mapped_column(JSON)
    note: Mapped[str] = mapped_column(String(1000), nullable=False, default='')
    route_results: Mapped[list['RouteResultRecord']] = relationship(
        back_populates='run', cascade='all, delete-orphan', passive_deletes=True)


class RouteResultRecord(Base):
    __tablename__ = 'route_results'
    __table_args__ = (
        CheckConstraint("route IN ('yolo','deepseek')", name='ck_route_results_route'),
        CheckConstraint("status IN ('ok','failed')", name='ck_route_results_status'),
        UniqueConstraint('run_id', 'route', name='uq_route_result_per_run'),
        Index('ix_route_results_run_id', 'run_id'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey('runs.run_id', ondelete='CASCADE'), nullable=False)
    route: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_dir: Mapped[str | None] = mapped_column(Text)
    run: Mapped[RunRecord] = relationship(back_populates='route_results')
    artifacts: Mapped[list['ArtifactRecord']] = relationship(
        back_populates='route_result', cascade='all, delete-orphan', passive_deletes=True)


class ArtifactRecord(Base):
    __tablename__ = 'artifacts'
    __table_args__ = (
        UniqueConstraint('route_result_id', 'name', name='uq_artifact_per_route_result'),
        Index('ix_artifacts_route_result_id', 'route_result_id'),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_result_id: Mapped[int] = mapped_column(
        ForeignKey('route_results.id', ondelete='CASCADE'), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    route_result: Mapped[RouteResultRecord] = relationship(back_populates='artifacts')


class SchemaVersion(Base):
    __tablename__ = 'schema_version'

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
