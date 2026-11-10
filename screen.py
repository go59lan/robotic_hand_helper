import asyncio
import os
import webbrowser
from pathlib import Path

from aiohttp import WSMsgType, web


UI_DIR = Path(__file__).parent / "ui"


class Screen:
    """Serves the child-facing page and pushes live events to it."""

    def __init__(self, port=None):
        self.port = port or int(os.getenv("HANDY_UI_PORT", "8080"))
        self.clients = set()
        self.messages = asyncio.Queue()
        # The latest value of each kind, so a page that (re)connects
        # immediately shows the current state instead of a blank screen.
        self.snapshot = {}
        self.runner = None
        self.broadcaster = None

    async def start(self):
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._websocket)
        app.router.add_static("/", UI_DIR)

        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "localhost", self.port).start()
        self.broadcaster = asyncio.create_task(self._broadcast())

        url = f"http://localhost:{self.port}"
        print(f"Screen: {url}")
        if os.getenv("HANDY_OPEN_BROWSER", "1") == "1":
            webbrowser.open(url)

    async def close(self):
        if not self.runner:
            return
        # Let the last messages (e.g. the goodbye state) reach the page.
        try:
            await asyncio.wait_for(self.messages.join(), timeout=1)
        except asyncio.TimeoutError:
            pass
        self.broadcaster.cancel()
        for ws in tuple(self.clients):
            await ws.close()
        await self.runner.cleanup()
        self.runner = None

    def send(self, kind, value):
        """Queue an event for every connected page. Safe to call often."""
        if kind != "level":  # mic level is too short-lived to replay
            self.snapshot[kind] = value
        self.messages.put_nowait({"type": kind, "value": value})

    async def _index(self, _request):
        return web.FileResponse(UI_DIR / "index.html")

    async def _websocket(self, request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        for kind, value in self.snapshot.items():
            await ws.send_json({"type": kind, "value": value})
        self.clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self.clients.discard(ws)
        return ws

    async def _broadcast(self):
        # A single sender keeps events in order.
        while True:
            message = await self.messages.get()
            for ws in tuple(self.clients):
                try:
                    await ws.send_json(message)
                except (ConnectionError, RuntimeError):
                    self.clients.discard(ws)
            self.messages.task_done()
