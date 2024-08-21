"""hot reloading"""

import asyncio
import functools
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType
from typing import Any, Awaitable, Callable, Iterator, MutableMapping

import watchdog
import watchdog.events
import watchdog.observers


class JustPretendItsAFuckingDict(MutableMapping[str, Any]):
    def __init__(self, obj: type) -> None:
        self.obj = obj

    def __getitem__(self, key: str) -> Any:
        return getattr(self.obj, key)

    def __setitem__(self, key: str, value: Any):
        return setattr(self.obj, key, value)

    def __delitem__(self, key: str) -> None:
        return delattr(self.obj, key)

    def __iter__(self) -> Iterator[str]:
        return self.obj.__dict__.__iter__()

    def __len__(self) -> int:
        return self.obj.__dict__.__len__()

    def __contains__(self, key: object) -> bool:
        # performance more?
        return self.obj.__dict__.__contains__(key)


class SexyReloader:
    def __init__(self, global_provider: Callable[[], dict[str, Any]],
                 loop: asyncio.AbstractEventLoop) -> None:
        self._globals = global_provider
        self._loop = loop
        self._observer = watchdog.observers.Observer()
        # root = Path(__file__).parent
        # self._state:  dict[Path, int] = {}
        # for file in root.rglob("*.py"):
        #     if file.name.startswith('x'):
        #         continue
        #     self._state[file] = file.stat().st_mtime_ns

    # def poll(self) -> set[str]:
    #     # IMPORTANT: this function is blocking and should be run IN
    #     # the main event loop to ensure that no other code is running
    #     # that the hotreload is modifying
    #     result = set()
    #     for file, last_time in self._state.items():
    #         modify_time = file.stat().st_mtime_ns
    #         if modify_time == last_time:
    #             continue
    #         self._state[file] = modify_time
    #         try:
    #             result |= self.reload(file)
    #         except Exception as e:
    #             print(f'Errored during hotreload: {e}')
    #     return result

    class _EventWatcher(watchdog.events.FileSystemEventHandler):
        def __init__(self, reloader: 'SexyReloader',
                     callback: Callable[[Path], Any]) -> None:
            super().__init__()
            self.callback = callback

        def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
            if isinstance(event, watchdog.events.FileModifiedEvent):
                self.callback(Path(event.src_path))
            return super().on_modified(event)

    def _handle_watch(self, callback: Callable[[set[str]], None], path: Path):
        print(f"Got update in {path}")
        if path.name.startswith('x'):
            return
        self._loop.call_soon_threadsafe(lambda: callback(self.reload(path)))

    def watch(self, callback: Callable[[set[str]], None]):
        root = Path(__file__).parent
        listener = SexyReloader._EventWatcher(
            self, functools.partial(self._handle_watch, callback))
        self._observer.schedule(
            listener, root, recursive=True,
            event_filter=(watchdog.events.FileModifiedEvent,))
        if not self._observer.is_alive():
            self._observer.start()

    def reload(self, file: Path) -> set[str]:
        old_globals = self._globals()
        mod_globals = old_globals.copy()
        # Dangerous!!! ^-^
        # pylint: disable-next=exec-used
        exec(file.read_text(encoding='utf8'), mod_globals)
        result = set()
        self._reload_value(old_globals, mod_globals, result)
        return result

    @staticmethod
    def compare_func(a: Callable, b: Callable,
                     *, _props=['co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_stacksize', 'co_flags', 'co_nlocals', 'co_consts', 'co_names', 'co_name', 'co_qualname', 'co_exceptiontable', 'co_varnames', 'co_cellvars', 'co_freevars', 'co_code']):
        for name in _props:
            if getattr(a.__code__, name) != getattr(b.__code__, name):
                print(f'{name} in {a.__qualname__}')
                return False
        return True

    def _reload_value(
            self,
            old: MutableMapping[str, Any],
            new: MutableMapping[str, Any],
            mutated: set[str],
            prefix: str = ''):
        for k, v in new.items():
            if k.startswith('__'):
                continue
            if k not in old:
                # New values and CONSTANTS
                old[k] = v
                mutated.add(f'{prefix}{k}')
                continue
            if old[k] == v:
                continue
            if k == k.upper():
                old[k] = v
                mutated.add(f'{prefix}{k}')
                continue
            match v:
                case type():
                    # Update class
                    self._reload_value(
                        JustPretendItsAFuckingDict(old[k]),
                        JustPretendItsAFuckingDict(v),
                        mutated, f'{prefix}{k}.')
                case FunctionType() | classmethod() | staticmethod():
                    if self.compare_func(v, old[k]):  # type: ignore
                        continue
                    old[k] = v
                    mutated.add(f'{prefix}{k}')
                case _:
                    pass
