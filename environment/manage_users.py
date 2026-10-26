"""Initialize local auth secrets and manage pre-created demonstration users."""

import argparse
import getpass
import json
import os
from pathlib import Path
import secrets
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deskvision import auth
from deskvision.database import get_database
from deskvision.models import LoginSession, UserRecord
from sqlalchemy import select


def init_local() -> int:
    path = ROOT / '.local/auth.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print('Auth configuration already exists; left unchanged.')
        return 0
    payload = {'jwt_secret': secrets.token_urlsafe(48), 'local_http': True}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream)
        stream.write('\n')
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    auth.settings.cache_clear()
    get_database().initialize()
    print('Local auth secret created; it is ignored by Git and was not printed.')
    return 0


def create_user(username: str) -> int:
    auth.settings()
    first = getpass.getpass('Set password (12-128 characters): ')
    second = getpass.getpass('Confirm password: ')
    if first != second:
        raise ValueError('Passwords did not match.')
    user_id = auth.create_user(username, first)
    print(f'Created account {username} ({user_id}). Keep the password private.')
    return 0


def create_demo_users() -> int:
    """Provision two random local-only credentials without echoing passwords."""
    auth.settings()
    path = ROOT / '.local/demo-accounts.json'
    if path.exists():
        print('Demo credential file already exists; left unchanged: .local/demo-accounts.json')
        return 0
    accounts = [{'username': name, 'password': secrets.token_urlsafe(18)}
                for name in ('teacher01', 'student01')]
    database = get_database()
    database.initialize()
    created_file = False
    try:
        with database.sessions.begin() as session:
            for account in accounts:
                if session.scalar(select(UserRecord).where(UserRecord.username == account['username'])):
                    raise ValueError('A demo username exists already. Use create-user with another name.')
                session.add(UserRecord(id=str(uuid4()), username=account['username'],
                                       password_hash=auth.PASSWORDS.hash(account['password'])))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x', encoding='utf-8') as stream:
                created_file = True
                json.dump({'purpose': 'Local demonstration only; keep private and reset before sharing.',
                           'accounts': accounts}, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.chmod(path, 0o600)
    except Exception:
        if created_file:
            path.unlink(missing_ok=True)
        raise
    print('Created teacher01 and student01. Read passwords locally in .local/demo-accounts.json; never commit it.')
    return 0


def disable_user(username: str) -> int:
    database = get_database()
    database.initialize()
    with database.sessions.begin() as session:
        user = session.scalar(select(UserRecord).where(UserRecord.username == username))
        if user is None:
            raise ValueError('Account not found.')
        user.active = False
        grants = session.scalars(select(LoginSession).where(LoginSession.user_id == user.id)).all()
        for grant in grants:
            grant.revoked = True
    print(f'Disabled account {username} and revoked its active sessions.')
    return 0


def set_password(username: str) -> int:
    auth.settings()
    first = getpass.getpass('Set new password (12-128 characters): ')
    second = getpass.getpass('Confirm new password: ')
    if first != second:
        raise ValueError('Passwords did not match.')
    if not 12 <= len(first) <= 128:
        raise ValueError('Password must contain 12-128 characters.')
    database = get_database()
    database.initialize()
    with database.sessions.begin() as session:
        user = session.scalar(select(UserRecord).where(UserRecord.username == username))
        if user is None:
            raise ValueError('Account not found.')
        from deskvision.auth import PASSWORDS
        user.password_hash = PASSWORDS.hash(first)
        user.active = True
        grants = session.scalars(select(LoginSession).where(LoginSession.user_id == user.id)).all()
        for grant in grants:
            grant.revoked = True
    print(f'Updated password for {username} and revoked previous sessions.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init-local', help='create a private local JWT signing secret')
    commands.add_parser('create-demo-users', help='create two local demo users with private random passwords')
    create = commands.add_parser('create-user', help='create a pre-provisioned demo account')
    create.add_argument('username')
    disable = commands.add_parser('disable-user', help='disable an account and revoke its sessions')
    disable.add_argument('username')
    reset = commands.add_parser('set-password', help='set a password and revoke existing sessions')
    reset.add_argument('username')
    args = parser.parse_args()
    try:
        if args.command == 'init-local':
            return init_local()
        if args.command == 'create-demo-users':
            return create_demo_users()
        if args.command == 'create-user':
            return create_user(args.username)
        if args.command == 'disable-user':
            return disable_user(args.username)
        return set_password(args.username)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
