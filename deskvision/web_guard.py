"""Bind Gradio files, sessions and queue events to the authenticated login.

This process-local grant index intentionally expires on restart: re-upload files
and open a new page after restarting. Persistent results use owner-checked API URLs.
"""

import json
from pathlib import Path
import re
from time import time
from urllib.parse import unquote, urlsplit

from fastapi import HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from . import auth


class WorkbenchGuard:
    def __init__(self, app):
        self.app = app
        self.grants = {}

    def _bucket(self, identity):
        now = time()
        for key in list(self.grants):
            if self.grants[key]['expires'] < now:
                del self.grants[key]
        return self.grants.setdefault(identity.session_id, {
            'expires': now + auth.TOKEN_SECONDS, 'files': set(),
            'session_hash': set(), 'event_id': set(), 'upload_id': set(),
        })

    def _bind(self, bucket, kind, value, *, create=False):
        if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', value):
            raise HTTPException(403, '无效的工作台会话。')
        if value in bucket[kind]:
            return
        if (not create or any(value in other[kind] for other in self.grants.values())
                or len(bucket[kind]) >= 4096):
            raise HTTPException(403, '工作台会话不属于当前登录。')
        bucket[kind].add(value)

    @staticmethod
    def _path(value):
        if not isinstance(value, str) or not value or value.startswith(('http:', 'https:', '//')):
            raise HTTPException(403, '只接受当前账号上传的文件。')
        return str(Path(value).resolve())

    def _check_files(self, value, bucket):
        if isinstance(value, dict):
            if 'path' in value:
                if self._path(value['path']) not in bucket['files']:
                    raise HTTPException(403, '文件不属于当前登录，请重新上传。')
                url = value.get('url')
                if url and (url.startswith(('http:', 'https:', '//')) or 'file=' not in url):
                    raise HTTPException(403, '不接受远程文件地址。')
            for child in value.values():
                self._check_files(child, bucket)
        elif isinstance(value, list):
            for child in value:
                self._check_files(child, bucket)

    def _capture(self, value, bucket, *, upload=False):
        if upload and isinstance(value, list):
            bucket['files'].update(self._path(path) for path in value if isinstance(path, str))
        if isinstance(value, dict):
            meta = value.get('meta')
            if (isinstance(meta, dict) and meta.get('_type') == 'gradio.FileData'
                    and value.get('path') and not value['path'].startswith(('http:', 'https:', '//'))):
                bucket['files'].add(self._path(value['path']))
            if isinstance(value.get('event_id'), str):
                self._bind(bucket, 'event_id', value['event_id'], create=True)
            for child in value.values():
                self._capture(child, bucket)
        elif isinstance(value, list):
            for child in value:
                self._capture(child, bucket)

    async def __call__(self, scope, receive, send):
        path = scope.get('path', '')
        if scope['type'] == 'websocket' and path.startswith('/ui'):
            await send({'type': 'websocket.close', 'code': 1008})
            return
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        ui = path == '/ui' or path.startswith('/ui/')
        try:
            # HTTP exceptions are returned here, outside FastAPI's route handlers.
            if path != '/health' and auth.settings()[1] and scope.get('scheme') != 'https':
                if request.client is None or request.client.host not in ('127.0.0.1', '::1'):
                    raise HTTPException(403, '本机 HTTP 模式仅允许回环地址；远程部署须启用 HTTPS。')
            if request.method not in ('GET', 'HEAD', 'OPTIONS'):
                origin = request.headers.get('origin')
                expected = f'{request.url.scheme}://{request.url.netloc}'
                if origin and origin != expected:
                    raise HTTPException(403, '不接受跨站修改请求。')
                if request.headers.get('sec-fetch-site') == 'cross-site':
                    raise HTTPException(403, '不接受跨站修改请求。')
                if (request.cookies.get(auth.COOKIE) and not request.headers.get('authorization')
                        and not origin):
                    raise HTTPException(403, 'Cookie 修改请求须带同源 Origin。')
            if not ui:
                await self.app(scope, receive, send)
                return
            identity = await run_in_threadpool(auth.current_user, request)
            bucket = self._bucket(identity)
            relative = path.removeprefix('/ui').removeprefix('/gradio_api').strip('/')
            # Unused proxy/stream/component-server routes can bypass file or event checks.
            if relative.startswith(('proxy=', 'stream/', 'component_server', 'reset',
                                    'monitoring', 'dev/', 'file/', 'deep_link/')):
                raise HTTPException(403, '此工作台不开放该接口。')
            if relative.startswith('file='):
                filename = self._path(unquote(relative[5:]))
                if filename not in bucket['files']:
                    raise HTTPException(404, '文件不存在。')
            for kind in ('session_hash', 'event_id', 'upload_id'):
                value = request.query_params.get(kind)
                if value:
                    self._bind(bucket, kind, value, create=kind == 'upload_id' and relative == 'upload')
            if relative.startswith('heartbeat/'):
                self._bind(bucket, 'session_hash', relative.split('/')[1], create=True)
            if relative.startswith('call/') and request.method == 'GET':
                self._bind(bucket, 'event_id', relative.split('/')[-1])
            messages = []
            if request.method == 'POST' and 'application/json' in request.headers.get('content-type', ''):
                body = bytearray()
                while True:
                    message = await receive()
                    messages.append(message)
                    body.extend(message.get('body', b''))
                    if len(body) > 4 * 1024 * 1024:
                        raise HTTPException(413, '工作台请求过大。')
                    if not message.get('more_body'):
                        break
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError):
                    raise HTTPException(400, '无效 JSON。') from None
                self._check_files(payload, bucket)
                if isinstance(payload, dict):
                    for kind in ('session_hash', 'event_id'):
                        if payload.get(kind):
                            self._bind(bucket, kind, payload[kind],
                                       create=kind == 'session_hash' and relative != 'cancel')

            async def replay():
                return messages.pop(0) if messages else await receive()

            content_type = ''
            response_status = 0
            pending = bytearray()
            closed = False

            async def guarded_send(message):
                nonlocal content_type, response_status, closed
                if closed:
                    return
                if message['type'] == 'http.response.start':
                    response_status = message['status']
                    headers = dict(message['headers'])
                    content_type = headers.get(b'content-type', b'').decode()
                    # Shared browser/proxy caches must never retain account data.
                    message['headers'] = [(key, val) for key, val in message['headers']
                                          if key.lower() != b'cache-control'] + [(b'cache-control', b'no-store')]
                if message['type'] == 'http.response.body' and response_status < 400:
                    if 'text/event-stream' in content_type:
                        try:
                            await run_in_threadpool(auth.current_user, request)
                        except HTTPException:
                            await send({'type': 'http.response.body', 'body': b'', 'more_body': False})
                            closed = True
                            return
                        pending.extend(message.get('body', b''))
                        while b'\n\n' in pending:
                            frame, _, rest = pending.partition(b'\n\n')
                            pending[:] = rest
                            for line in frame.splitlines():
                                if line.startswith(b'data:'):
                                    try:
                                        self._capture(json.loads(line[5:]), bucket)
                                    except (ValueError, UnicodeError):
                                        pass
                            await send({'type': 'http.response.body', 'body': bytes(frame) + b'\n\n',
                                        'more_body': True})
                        if message.get('more_body'):
                            return
                        message = {**message, 'body': bytes(pending)}
                    elif ('application/json' in content_type
                          and relative not in ('config', 'info', 'openapi.json')):
                        pending.extend(message.get('body', b''))
                        if not message.get('more_body'):
                            try:
                                self._capture(json.loads(pending), bucket, upload=relative == 'upload')
                            except (ValueError, UnicodeError):
                                pass
                await send(message)

            await self.app(scope, replay, guarded_send)
        except HTTPException as error:
            await JSONResponse({'detail': error.detail}, status_code=error.status_code,
                               headers=error.headers)(scope, receive, send)
