"""Shared JWT authentication for HTTP, browser cookies and Gradio callbacks."""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import secrets
from time import time
from uuid import uuid4

from fastapi import HTTPException, Request
import jwt
from pwdlib import PasswordHash
from sqlalchemy import select

from .config import ROOT
from .database import get_database
from .models import LoginSession, UserRecord


COOKIE = 'deskvision_session'
PASSWORDS = PasswordHash.recommended()
DUMMY_HASH = PASSWORDS.hash(secrets.token_urlsafe(24))
TOKEN_SECONDS = 1800


@dataclass(frozen=True)
class Identity:
    user_id: str
    username: str
    session_id: str


@lru_cache(maxsize=1)
def settings() -> tuple[str, bool]:
    """Secrets are provisioned explicitly, never generated silently at startup."""
    config_path = Path(os.environ.get('DESKVISION_AUTH_CONFIG', ROOT / '.local/auth.json'))
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    secret = os.environ.get('DESKVISION_JWT_SECRET', config.get('jwt_secret', ''))
    local_http = os.environ.get('DESKVISION_LOCAL_HTTP', str(config.get('local_http', False))).lower() in ('1', 'true')
    if not isinstance(secret, str) or len(secret.encode()) < 32:
        raise RuntimeError('缺少至少 32 字节 JWT 密钥。先运行 environment/manage_users.py init-local。')
    return secret, local_http


def database():
    db = get_database()
    db.initialize()
    return db


def create_user(username: str, password: str) -> str:
    if not re.fullmatch(r'[a-zA-Z0-9_\-]{3,32}', username):
        raise ValueError('账号须为 3～32 位字母、数字、下划线或连字符。')
    if not 12 <= len(password) <= 128:
        raise ValueError('密码须为 12～128 个字符。')
    with database().sessions.begin() as session:
        if session.scalar(select(UserRecord).where(UserRecord.username == username)):
            raise ValueError('账号已存在。')
        user_id = str(uuid4())
        session.add(UserRecord(id=user_id, username=username, password_hash=PASSWORDS.hash(password)))
    return user_id


def login(username: str, password: str) -> tuple[str, Identity]:
    with database().sessions.begin() as session:
        user = session.scalar(select(UserRecord).where(UserRecord.username == username))
        valid = PASSWORDS.verify(password, user.password_hash if user else DUMMY_HASH)
        if not valid or user is None or not user.active:
            raise HTTPException(401, '账号或密码错误。')
        issued = int(time())
        session_id = str(uuid4())
        session.add(LoginSession(id=session_id, user_id=user.id, expires_at=issued + TOKEN_SECONDS))
        token = jwt.encode({'sub': user.id, 'jti': session_id, 'iat': issued,
                            'exp': issued + TOKEN_SECONDS, 'iss': 'deskvision',
                            'aud': 'deskvision'}, settings()[0], algorithm='HS256')
        return token, Identity(user.id, user.username, session_id)


def resolve_token(token: str) -> Identity:
    try:
        claims = jwt.decode(token, settings()[0], algorithms=['HS256'], issuer='deskvision',
                            audience='deskvision', options={'require': ['sub', 'jti', 'iat', 'exp']})
        with database().sessions() as session:
            user = session.get(UserRecord, claims['sub'])
            grant = session.get(LoginSession, claims['jti'])
            if (user is None or not user.active or grant is None or grant.revoked
                    or grant.user_id != user.id or grant.expires_at <= int(time())):
                raise ValueError('revoked or inactive')
            return Identity(user.id, user.username, grant.id)
    except (jwt.InvalidTokenError, ValueError, TypeError, KeyError):
        raise HTTPException(401, '登录已失效，请重新登录。', headers={'WWW-Authenticate': 'Bearer'}) from None


def token_from_request(request: Request) -> str | None:
    authorization = request.headers.get('authorization', '')
    if authorization:
        scheme, _, token = authorization.partition(' ')
        if scheme.lower() != 'bearer' or not token:
            raise HTTPException(401, '需要 Bearer 登录凭据。')
        return token
    return request.cookies.get(COOKIE)


def current_user(request: Request) -> Identity:
    token = token_from_request(request)
    if not token:
        raise HTTPException(401, '请先登录。', headers={'WWW-Authenticate': 'Bearer'})
    return resolve_token(token)


def revoke(identity: Identity) -> None:
    with database().sessions.begin() as session:
        grant = session.get(LoginSession, identity.session_id)
        if grant:
            grant.revoked = True


def gradio_user(request: Request) -> str | None:
    try:
        return current_user(request).user_id
    except HTTPException:
        return None


def callback_user(request) -> Identity:
    """Revalidate a queued callback; neither client JSON nor gr.State is authority."""
    if request is None or request.request is None:
        raise HTTPException(401, '请从登录后的工作台操作。')
    return current_user(request.request)
