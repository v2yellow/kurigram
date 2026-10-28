#  Pyrogram - Telegram MTProto API Client Library for Python
#  Copyright (C) 2017-present Dan <https://github.com/delivrance>
#
#  This file is part of Pyrogram.
#
#  Pyrogram is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Pyrogram is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with Pyrogram.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations as _annotations

import asyncio
import contextlib
import io
from typing import TYPE_CHECKING, Final, cast

import pytest

import pyrogram
from pyrogram import StopTransmission, raw, types
from pyrogram.errors import FloodWait
from pyrogram.methods.advanced.save_file import SaveFile

if TYPE_CHECKING:
    from concurrent.futures import Executor
    from pathlib import Path

_PART_SIZE: Final[int] = 512 * 1024

# `rnd_id()` below answers with this too, so a re-upload addresses the same file as the
#  upload that produced it.
_FILE_ID: Final[int] = 1234567890


class Storage:
    async def dc_id(self) -> int:
        return 2


class Media:
    """A media session that records the parts it was asked to save."""

    def __init__(self, rejects_part: int | None = None) -> None:
        self.rejects_part = rejects_part
        self.saved_parts: list[int] = []

    async def invoke(
        self,
        query: raw.functions.upload.SaveFilePart | raw.functions.upload.SaveBigFilePart,
    ) -> None:
        if query.file_part == self.rejects_part:
            raise ConnectionError("the server closed the connection")

        self.saved_parts.append(query.file_part)


class Uploader(SaveFile):
    def __init__(self, media: Media) -> None:
        self.media = media
        self.name = "uploader"
        self.me: types.User | None = None
        self.storage = Storage()
        self.executor: Executor | None = None
        self.save_file_semaphore = asyncio.Semaphore(1)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.get_event_loop()

    def rnd_id(self) -> int:
        return _FILE_ID

    async def get_session(self, dc_id: int, *, is_media: bool = False) -> Media:
        return self.media


def uploader(media: Media) -> pyrogram.Client:
    # `Uploader` subclasses only the mixin, and the `save_file` overloads pin
    #  `self: pyrogram.Client`.
    return cast("pyrogram.Client", Uploader(media))


@pytest.fixture
def three_parts(tmp_path: Path) -> str:
    path = tmp_path / "upload.bin"
    path.write_bytes(b"x" * (2 * _PART_SIZE + 1))

    return str(path)


@pytest.mark.asyncio
async def test_a_finished_upload_describes_every_part(three_parts: str) -> None:
    media = Media()

    file = await uploader(media).save_file(three_parts)

    assert isinstance(file, raw.types.InputFile)
    assert file.parts == 3
    assert media.saved_parts == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_part_the_server_refused_reaches_the_caller(three_parts: str) -> None:
    # The failure used to be logged inside the worker and nowhere else: `save_file()` handed back
    #  an `InputFile` for a file the server never received in full, and the send that followed
    #  failed with an unrelated error (much later still, for a caller that stored the id).
    media = Media(rejects_part=1)

    with pytest.raises(ConnectionError):
        await uploader(media).save_file(three_parts)


@pytest.mark.asyncio
async def test_a_failed_part_stops_the_upload_before_later_parts_are_sent(three_parts: str) -> None:
    media = Media(rejects_part=1)

    with pytest.raises(ConnectionError):
        await uploader(media).save_file(three_parts)

    # Cancellation wakes the producer even when its one-slot queue is full.
    assert media.saved_parts == [0]


@pytest.mark.asyncio
async def test_re_uploading_one_missing_part_answers_with_nothing(three_parts: str) -> None:
    media = Media()

    file = await uploader(media).save_file(
        three_parts,
        file_id=_FILE_ID,
        file_part=1,
    )

    assert file is None
    assert media.saved_parts == [1]


@pytest.mark.asyncio
async def test_a_missing_part_the_server_refused_reaches_the_caller_too(three_parts: str) -> None:
    media = Media(rejects_part=1)

    with pytest.raises(ConnectionError):
        await uploader(media).save_file(
            three_parts,
            file_id=_FILE_ID,
            file_part=1,
        )


@pytest.mark.asyncio
async def test_no_path_is_not_an_upload_at_all() -> None:
    assert await uploader(Media()).save_file(None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file",
    [
        raw.types.InputFile(id=_FILE_ID, parts=1, name="cached.bin", md5_checksum="hash"),
        raw.types.InputFileBig(id=_FILE_ID, parts=1, name="cached.bin"),
    ],
)
async def test_an_existing_input_file_is_returned_without_upload(file) -> None:
    media = Media()

    assert await uploader(media).save_file(file) is file
    assert media.saved_parts == []


class TrackedFile(io.BytesIO):
    name = "upload.bin"


class FloodingMedia(Media):
    def __init__(self) -> None:
        super().__init__()
        self.sent_parts: list[int] = []

    async def invoke(
        self,
        query: raw.functions.upload.SaveFilePart | raw.functions.upload.SaveBigFilePart,
    ) -> None:
        self.sent_parts.append(query.file_part)

        if len(self.sent_parts) == 1:
            raise FloodWait(3600)


class BlockingMedia(Media):
    def __init__(self, workers_count: int = 1) -> None:
        super().__init__()
        self.workers_count = workers_count
        self.sent_parts: list[int] = []
        self.workers_started = asyncio.Event()
        self.workers_cancelled = asyncio.Event()
        self._cancelled_parts: list[int] = []
        self.release = asyncio.Event()

    async def invoke(
        self,
        query: raw.functions.upload.SaveFilePart | raw.functions.upload.SaveBigFilePart,
    ) -> None:
        self.sent_parts.append(query.file_part)

        if len(self.sent_parts) == self.workers_count:
            self.workers_started.set()

        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self._cancelled_parts.append(query.file_part)

            if len(self._cancelled_parts) == self.workers_count:
                self.workers_cancelled.set()

            raise


async def await_cancelled(task: asyncio.Task[object]) -> None:
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=0.2)


@pytest.mark.asyncio
async def test_a_long_flood_wait_retries_when_the_upload_is_not_cancelled(monkeypatch) -> None:
    media = FloodingMedia()
    tracked_file = TrackedFile(b"x")
    waits: list[int] = []

    async def skip_wait(seconds: int) -> None:
        waits.append(seconds)

    monkeypatch.setattr("pyrogram.methods.advanced.save_file.asyncio.sleep", skip_wait)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: tracked_file)

    file = await uploader(media).save_file("upload.bin")

    assert isinstance(file, raw.types.InputFile)
    assert waits == [3600]
    assert media.sent_parts == [0, 0]
    assert tracked_file.closed


@pytest.mark.asyncio
async def test_cancelling_a_long_flood_wait_stops_the_worker_and_closes_the_file(
    monkeypatch,
) -> None:
    media = FloodingMedia()
    tracked_file = TrackedFile(b"x")
    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()
    sleep_cancelled = asyncio.Event()

    async def held_sleep(seconds: int) -> None:
        assert seconds == 3600
        sleep_started.set()

        try:
            await release_sleep.wait()
        except asyncio.CancelledError:
            sleep_cancelled.set()
            raise

    monkeypatch.setattr("pyrogram.methods.advanced.save_file.asyncio.sleep", held_sleep)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: tracked_file)

    upload = asyncio.create_task(uploader(media).save_file("upload.bin"))

    await sleep_started.wait()
    upload.cancel()

    try:
        await await_cancelled(upload)

        assert media.sent_parts == [0]
        assert sleep_cancelled.is_set()
        assert tracked_file.closed
    finally:
        release_sleep.set()

        if not upload.done():
            with contextlib.suppress(asyncio.CancelledError):
                await upload


@pytest.mark.asyncio
async def test_cancelling_an_active_rpc_keeps_the_caller_file_open() -> None:
    media = BlockingMedia()
    tracked_file = TrackedFile(b"x")

    upload = asyncio.create_task(uploader(media).save_file(tracked_file))

    await media.workers_started.wait()
    upload.cancel()

    try:
        await await_cancelled(upload)

        assert media.sent_parts == [0]
        assert media.workers_cancelled.is_set()
        assert not tracked_file.closed
    finally:
        media.release.set()

        if not upload.done():
            with contextlib.suppress(asyncio.CancelledError):
                await upload


@pytest.mark.asyncio
async def test_stop_transmission_cancels_every_active_upload_worker(monkeypatch) -> None:
    media = BlockingMedia(workers_count=4)
    tracked_file = TrackedFile(b"x" * (21 * _PART_SIZE))
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: tracked_file)

    async def stop_after_every_worker_is_active(*args: object) -> None:
        await asyncio.sleep(0)

        if media.workers_started.is_set():
            raise StopTransmission

    with pytest.raises(StopTransmission):
        await uploader(media).save_file("upload.bin", progress=stop_after_every_worker_is_active)

    assert len(media.sent_parts) == 4
    assert media.workers_cancelled.is_set()
    assert tracked_file.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "asynchronous",
    [
        pytest.param(True, id="async-callback"),
        pytest.param(False, id="sync-callback-via-executor"),
    ],
)
async def test_stop_transmission_from_a_progress_callback_ends_the_upload(
    three_parts: str,
    *,
    asynchronous: bool,
) -> None:
    media = Media()
    calls: list[int] = []

    def cancel_on_second_part(current: int, total: int) -> None:
        calls.append(current)

        if len(calls) == 2:
            raise StopTransmission

    async def asynchronous_cancel(current: int, total: int) -> None:
        cancel_on_second_part(current, total)

    progress = asynchronous_cancel if asynchronous else cancel_on_second_part

    with pytest.raises(StopTransmission):
        await uploader(media).save_file(three_parts, progress=progress)

    # Cancellation also stops a produced part that has not entered its RPC yet.
    # The third part is never produced, and no worker sends after the abort returns.
    assert media.saved_parts in ([0], [0, 1])
    saved_parts = media.saved_parts.copy()
    await asyncio.sleep(0)
    assert media.saved_parts == saved_parts


@pytest.mark.asyncio
async def test_failed_big_upload_cancels_blocked_siblings_without_draining_file() -> None:
    class FailedMedia(BlockingMedia):
        async def invoke(self, query) -> None:
            self.sent_parts.append(query.file_part)
            if len(self.sent_parts) == 4:
                self.workers_started.set()
            if query.file_part == 0:
                await self.workers_started.wait()
                raise ConnectionError("part retry budget exhausted")
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self._cancelled_parts.append(query.file_part)
                raise

    media = FailedMedia(workers_count=4)
    file = TrackedFile(b"x" * (32 * _PART_SIZE))
    client = uploader(media)
    with pytest.raises(ConnectionError, match="part retry budget exhausted"):
        await asyncio.wait_for(client.save_file(file), timeout=1)
    assert len(media.sent_parts) == 4
    assert len(media._cancelled_parts) == 3
    assert not file.closed
    assert not client.save_file_semaphore.locked()
    before = media.sent_parts.copy()
    await asyncio.sleep(0)
    assert media.sent_parts == before


@pytest.mark.asyncio
async def test_upload_without_acknowledged_progress_times_out_and_joins_worker(monkeypatch) -> None:
    monkeypatch.setattr(SaveFile, "UPLOAD_STALL_TIMEOUT", 0.05, raising=False)
    media = BlockingMedia()
    file = TrackedFile(b"x")
    client = uploader(media)
    with pytest.raises(TimeoutError, match="acknowledged progress"):
        await asyncio.wait_for(client.save_file(file), timeout=1)
    assert media.workers_cancelled.is_set()
    assert not client.save_file_semaphore.locked()
    assert not file.closed


@pytest.mark.asyncio
async def test_acknowledged_progress_allows_upload_longer_than_stall_budget(monkeypatch) -> None:
    monkeypatch.setattr(SaveFile, "UPLOAD_STALL_TIMEOUT", 0.2, raising=False)

    class SlowMedia(Media):
        async def invoke(self, query) -> None:
            await asyncio.sleep(0.04)
            await super().invoke(query)

    media = SlowMedia()
    file = TrackedFile(b"x" * (8 * _PART_SIZE))
    result = await asyncio.wait_for(uploader(media).save_file(file), timeout=2)
    assert isinstance(result, raw.types.InputFile)
    assert media.saved_parts == list(range(8))


@pytest.mark.asyncio
async def test_provider_flood_wait_is_not_mistaken_for_stalled_upload(monkeypatch) -> None:
    monkeypatch.setattr(SaveFile, "UPLOAD_STALL_TIMEOUT", 0.05, raising=False)

    class RateLimitedMedia(Media):
        attempts = 0

        async def invoke(self, query) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise FloodWait(1)
            await super().invoke(query)

    media = RateLimitedMedia()
    result = await asyncio.wait_for(uploader(media).save_file(TrackedFile(b"x")), timeout=2)
    assert isinstance(result, raw.types.InputFile)
    assert media.attempts == 2
    assert media.saved_parts == [0]


@pytest.mark.asyncio
async def test_callback_failure_retires_watchdog_even_if_wait_for_consumes_cancellation(
    monkeypatch,
) -> None:
    wait_for = asyncio.wait_for
    watching = asyncio.Event()

    async def consume_watcher_cancel(awaitable, timeout):
        if timeout is not None and timeout > 100:
            watching.set()
            try:
                return await wait_for(awaitable, timeout)
            except asyncio.CancelledError:
                # Reproduce the Python 3.10 completed-event cancellation race.
                return True
        return await wait_for(awaitable, timeout)

    async def failed_callback(*args):
        await watching.wait()
        raise RuntimeError("progress callback aborted")

    monkeypatch.setattr(
        "pyrogram.methods.advanced.save_file.asyncio.wait_for", consume_watcher_cancel
    )
    client = uploader(Media())
    with pytest.raises(RuntimeError, match="progress callback aborted"):
        await wait_for(client.save_file(TrackedFile(b"x"), progress=failed_callback), timeout=1)
    assert not client.save_file_semaphore.locked()


@pytest.mark.asyncio
async def test_stalled_progress_callback_is_cancelled_before_upload_returns(monkeypatch) -> None:
    monkeypatch.setattr(SaveFile, "UPLOAD_STALL_TIMEOUT", 0.05)
    cancelled = asyncio.Event()
    hold = asyncio.Event()

    async def stalled_callback(*args):
        try:
            await hold.wait()
        finally:
            cancelled.set()

    media = Media()
    with pytest.raises(TimeoutError, match="acknowledged progress"):
        await asyncio.wait_for(
            uploader(media).save_file(TrackedFile(b"x"), progress=stalled_callback), timeout=1
        )
    assert cancelled.is_set()
    assert media.saved_parts == [0]


@pytest.mark.asyncio
async def test_cancelled_rpc_cannot_resume_worker_after_provider_swallows_cancellation() -> None:
    class SwallowingMedia(BlockingMedia):
        async def invoke(self, query):
            self.sent_parts.append(query.file_part)
            if len(self.sent_parts) == 4:
                self.workers_started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                return

    media = SwallowingMedia(workers_count=4)
    upload = asyncio.create_task(uploader(media).save_file(TrackedFile(b"x" * (32 * _PART_SIZE))))
    await media.workers_started.wait()
    upload.cancel()
    try:
        await await_cancelled(upload)
        assert len(media.sent_parts) == 4
    finally:
        # A regressed worker can be waiting on its queue after consuming cancellation.
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and ".worker" in task.get_coro().__qualname__:
                task.cancel()
        media.release.set()
        if not upload.done():
            upload.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upload


@pytest.mark.asyncio
async def test_cancelled_callback_cannot_resume_producer_after_consuming_cancellation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def swallowing_callback(*args):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            return

    media = Media()
    upload = asyncio.create_task(
        uploader(media).save_file(
            TrackedFile(b"x" * (8 * _PART_SIZE)), progress=swallowing_callback
        )
    )
    await started.wait()
    upload.cancel()
    try:
        await await_cancelled(upload)
        before = media.saved_parts.copy()
        await asyncio.sleep(0)
        assert media.saved_parts == before
    finally:
        release.set()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and ".produce" in task.get_coro().__qualname__:
                task.cancel()
        if not upload.done():
            upload.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upload


@pytest.mark.asyncio
async def test_cancellation_consumed_as_flood_wait_cannot_hold_upload_cleanup() -> None:
    class SwallowingFloodMedia(BlockingMedia):
        async def invoke(self, query):
            self.workers_started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                raise FloodWait(3600) from None

    media = SwallowingFloodMedia()
    upload = asyncio.create_task(uploader(media).save_file(TrackedFile(b"x")))
    await media.workers_started.wait()
    upload.cancel()
    try:
        await await_cancelled(upload)
    finally:
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and ".worker" in task.get_coro().__qualname__:
                task.cancel()
        media.release.set()
        if not upload.done():
            upload.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upload


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["part", "sentinel"])
async def test_consumed_queue_cancellation_cannot_resume_producer(monkeypatch, when) -> None:
    queue_type = asyncio.Queue
    entered = asyncio.Event()
    hold = asyncio.Event()
    callbacks = []

    class SwallowingQueue(queue_type):
        async def put(self, item):
            if not entered.is_set() and ((item is None) == (when == "sentinel")):
                entered.set()
                try:
                    await hold.wait()
                except asyncio.CancelledError:
                    return
            return await super().put(item)

    async def progress(*args):
        callbacks.append(args)

    monkeypatch.setattr("pyrogram.methods.advanced.save_file.asyncio.Queue", SwallowingQueue)
    media = Media()
    upload = asyncio.create_task(
        uploader(media).save_file(TrackedFile(b"x" * (32 * _PART_SIZE)), progress=progress)
    )
    await entered.wait()
    before = len(callbacks)
    upload.cancel()
    try:
        await await_cancelled(upload)
        assert len(callbacks) == before
    finally:
        hold.set()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and ".produce" in task.get_coro().__qualname__:
                task.cancel()
        if not upload.done():
            upload.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await upload
