from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import shutil
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TextIO

from psycopg.conninfo import conninfo_to_dict, make_conninfo


DEDICATED_DATABASE_PATTERN = re.compile(r"^vvb001_e2e_[a-z0-9_]{1,48}$")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
INITDB_TIMEOUT_SECONDS = 120
PG_CTL_LAUNCH_TIMEOUT_SECONDS = 15
STARTUP_TIMEOUT_SECONDS = 60
PROBE_TIMEOUT_SECONDS = 5
SHUTDOWN_TIMEOUT_SECONDS = 30
READINESS_POLL_SECONDS = 0.25


@dataclass(frozen=True)
class NativePostgresBinaries:
    bin_dir: Path
    postgres: Path
    pg_ctl: Path
    initdb: Path
    pg_isready: Path
    psql: Path | None


class NativePostgresLifecycleError(RuntimeError):
    """Bounded, diagnostic failure for a disposable native PostgreSQL cluster."""


def _tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return "(log file not created)"
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read log: {exc})"
    return "\n".join(content[-lines:]) or "(log file empty)"


def _diagnostic(
    operation: str,
    *,
    timeout: float | None,
    data_dir: Path,
    host: str,
    port: int,
    log_path: Path,
    detail: str,
) -> NativePostgresLifecycleError:
    timeout_line = f"\nTimeout: {timeout:g} seconds" if timeout is not None else ""
    return NativePostgresLifecycleError(
        f"{operation} failed.{timeout_line}\nHost: {host}\nPort: {port}"
        f"\nData directory: {data_dir}\nLog: {log_path}\n{detail}"
        f"\nLast PostgreSQL log lines:\n{_tail(log_path)}"
    )


def port_is_occupied(host: str, port: int, *, timeout: float = 0.25) -> bool:
    if not 1 <= port <= 65535:
        raise ValueError("PostgreSQL E2E port must be between 1 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def run_checked_command(
    command: list[str],
    *,
    operation: str,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Run a finite utility command; never include a credential in command arguments."""
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativePostgresLifecycleError(
            f"{operation} timed out after {timeout:g} seconds ({Path(command[0]).name})"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()
        )
        raise NativePostgresLifecycleError(
            f"{operation} failed ({Path(command[0]).name}, exit {exc.returncode})"
            + (f":\n{details}" if details else "")
        ) from exc


def launch_postgres(
    binaries: NativePostgresBinaries,
    *,
    data_dir: Path,
    log_path: Path,
    launcher_log_path: Path,
    host: str,
    port: int,
    timeout: float = PG_CTL_LAUNCH_TIMEOUT_SECONDS,
    popen_factory: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> None:
    """Launch through pg_ctl without waiting for the foreground server lifetime.

    Output is redirected to a regular file, not PIPE. This is important on Windows because the
    detached cmd/postgres process tree may retain inherited handles after pg_ctl exits.
    """
    command = [
        str(binaries.pg_ctl),
        "-D", str(data_dir),
        "-l", str(log_path),
        "-o", f"-p {port} -h {host}",
        "-W",
        "start",
    ]
    launcher_log_path.parent.mkdir(parents=True, exist_ok=True)
    with launcher_log_path.open("ab", buffering=0) as launcher_output:
        process = popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=launcher_output,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            with suppress(Exception):
                process.kill()
            with suppress(Exception):
                process.wait(timeout=5)
            raise _diagnostic(
                "PostgreSQL launcher",
                timeout=timeout,
                data_dir=data_dir,
                host=host,
                port=port,
                log_path=log_path,
                detail=f"pg_ctl did not return. Launcher log: {launcher_log_path}\n{_tail(launcher_log_path)}",
            ) from exc
    if return_code != 0:
        raise _diagnostic(
            "PostgreSQL launcher",
            timeout=None,
            data_dir=data_dir,
            host=host,
            port=port,
            log_path=log_path,
            detail=f"pg_ctl exited with code {return_code}. Launcher log: {launcher_log_path}\n{_tail(launcher_log_path)}",
        )


def read_postmaster_pid(data_dir: Path) -> int | None:
    path = data_dir / "postmaster.pid"
    if not path.is_file():
        return None
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        return None


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_postgres(
    binaries: NativePostgresBinaries,
    *,
    data_dir: Path,
    log_path: Path,
    host: str,
    port: int,
    admin_user: str,
    timeout: float = STARTUP_TIMEOUT_SECONDS,
    poll_seconds: float = READINESS_POLL_SECONDS,
    run_probe: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    pid_exists: Callable[[int], bool] = process_exists,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    deadline = monotonic() + timeout
    owned_pid: int | None = None
    while True:
        try:
            result = run_probe(
                [
                    str(binaries.pg_isready), "-h", host, "-p", str(port),
                    "-d", "postgres", "-U", admin_user,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=PROBE_TIMEOUT_SECONDS,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess([], 2, "", "pg_isready timed out")
        owned_pid = read_postmaster_pid(data_dir) or owned_pid
        if result.returncode == 0:
            if owned_pid is None:
                raise _diagnostic(
                    "PostgreSQL readiness identity check",
                    timeout=None,
                    data_dir=data_dir,
                    host=host,
                    port=port,
                    log_path=log_path,
                    detail="pg_isready succeeded but the temporary cluster has no postmaster.pid",
                )
            return owned_pid
        if owned_pid is not None and not pid_exists(owned_pid):
            raise _diagnostic(
                "PostgreSQL startup",
                timeout=None,
                data_dir=data_dir,
                host=host,
                port=port,
                log_path=log_path,
                detail=f"Temporary PostgreSQL exited before readiness (owned PID {owned_pid}).",
            )
        if monotonic() >= deadline:
            probe_detail = "\n".join(
                value.strip() for value in (result.stdout, result.stderr) if value and value.strip()
            )
            raise _diagnostic(
                "PostgreSQL readiness",
                timeout=timeout,
                data_dir=data_dir,
                host=host,
                port=port,
                log_path=log_path,
                detail=f"pg_isready never reported ready. Last probe: {probe_detail or '(no output)'}",
            )
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))


def _wait_for_process_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not process_exists(pid)


def stop_owned_postgres(
    binaries: NativePostgresBinaries,
    *,
    data_dir: Path,
    log_path: Path,
    host: str,
    port: int,
    owned_pid: int | None,
    timeout: float = SHUTDOWN_TIMEOUT_SECONDS,
) -> None:
    current_pid = read_postmaster_pid(data_dir)
    if owned_pid is not None and current_pid is not None and current_pid != owned_pid:
        raise _diagnostic(
            "PostgreSQL cleanup ownership check",
            timeout=None,
            data_dir=data_dir,
            host=host,
            port=port,
            log_path=log_path,
            detail=f"Refusing cleanup: expected PID {owned_pid}, data directory reports PID {current_pid}.",
        )
    target_pid = owned_pid or current_pid
    if target_pid is None and not port_is_occupied(host, port):
        return
    try:
        run_checked_command(
            [
                str(binaries.pg_ctl), "-D", str(data_dir), "-m", "fast",
                "-w", "-t", str(int(timeout)), "stop",
            ],
            operation="PostgreSQL shutdown",
            timeout=timeout + 5,
        )
    except NativePostgresLifecycleError as stop_error:
        if target_pid is None:
            raise _diagnostic(
                "PostgreSQL shutdown",
                timeout=timeout,
                data_dir=data_dir,
                host=host,
                port=port,
                log_path=log_path,
                detail=str(stop_error),
            ) from stop_error
        # Exact owned postmaster only; never terminate PostgreSQL by process name.
        with suppress(ProcessLookupError):
            os.kill(target_pid, signal.SIGTERM)
        if not _wait_for_process_exit(target_pid, 10):
            raise _diagnostic(
                "PostgreSQL shutdown fallback",
                timeout=10,
                data_dir=data_dir,
                host=host,
                port=port,
                log_path=log_path,
                detail=f"Owned PID {target_pid} did not exit after exact-PID termination.",
            ) from stop_error
    deadline = time.monotonic() + 10
    while port_is_occupied(host, port) and time.monotonic() < deadline:
        time.sleep(0.1)
    if port_is_occupied(host, port):
        raise _diagnostic(
            "PostgreSQL port release",
            timeout=10,
            data_dir=data_dir,
            host=host,
            port=port,
            log_path=log_path,
            detail="The temporary cluster stopped but its requested port remains occupied.",
        )


@contextmanager
def temporary_native_postgres(
    binaries: NativePostgresBinaries,
    *,
    root: Path,
    host: str,
    port: int,
    admin_user: str,
    admin_password: str,
    progress: Callable[[str], None] = lambda _message: None,
) -> Iterator[tuple[Path, Path, int]]:
    data_dir = root / "data"
    log_path = root / "postgres.log"
    launcher_log_path = root / "pg_ctl_start.log"
    password_file = root / "admin-password.txt"
    owned_pid: int | None = None
    launch_attempted = False
    if port_is_occupied(host, port):
        raise NativePostgresLifecycleError(
            f"Cannot start temporary PostgreSQL: {host}:{port} is already in use. "
            "Choose another --port value or stop the conflicting process."
        )
    try:
        progress("Initializing PostgreSQL...")
        password_file.write_text(admin_password + "\n", encoding="utf-8")
        run_checked_command(
            [
                str(binaries.initdb), "-D", str(data_dir), "-U", admin_user,
                "--encoding=UTF8", "--auth-local=trust", "--auth-host=scram-sha-256",
                "--pwfile", str(password_file),
            ],
            operation="PostgreSQL initialization",
            timeout=INITDB_TIMEOUT_SECONDS,
        )
        password_file.unlink(missing_ok=True)
        progress(f"Starting PostgreSQL on {host}:{port}...")
        launch_attempted = True
        launch_postgres(
            binaries,
            data_dir=data_dir,
            log_path=log_path,
            launcher_log_path=launcher_log_path,
            host=host,
            port=port,
        )
        progress("Waiting for PostgreSQL readiness...")
        owned_pid = wait_for_postgres(
            binaries,
            data_dir=data_dir,
            log_path=log_path,
            host=host,
            port=port,
            admin_user=admin_user,
        )
        progress(f"PostgreSQL ready (owned PID {owned_pid}).")
        yield data_dir, log_path, owned_pid
    except BaseException as exc:
        if isinstance(exc, NativePostgresLifecycleError):
            progress(f"ERROR: {exc}")
        raise
    finally:
        password_file.unlink(missing_ok=True)
        current_pid = read_postmaster_pid(data_dir)
        if launch_attempted and (owned_pid is not None or current_pid is not None):
            progress("Stopping temporary PostgreSQL...")
            stop_owned_postgres(
                binaries,
                data_dir=data_dir,
                log_path=log_path,
                host=host,
                port=port,
                owned_pid=owned_pid or current_pid,
            )
            progress("Temporary PostgreSQL stopped and port released.")


def validate_dedicated_database_name(name: str) -> str:
    if not DEDICATED_DATABASE_PATTERN.fullmatch(name):
        raise ValueError(
            "native PostgreSQL E2E database must match "
            "vvb001_e2e_[a-z0-9_]{1,48}; refusing a broad or non-dedicated target"
        )
    return name


def validate_loopback_dsn(dsn: str) -> dict[str, str]:
    values = {key: str(value) for key, value in conninfo_to_dict(dsn).items() if value is not None}
    host = values.get("host", "").strip().lower()
    if host not in LOOPBACK_HOSTS:
        raise ValueError("native PostgreSQL E2E requires an explicit loopback host")
    if "," in host or " " in host:
        raise ValueError("multiple PostgreSQL hosts are not allowed for E2E")
    return values


def dsn_for_database(dsn: str, database: str, *, user: str | None = None, password: str | None = None) -> str:
    validate_dedicated_database_name(database)
    validate_loopback_dsn(dsn)
    updates: dict[str, str] = {"dbname": database}
    if user is not None:
        updates["user"] = user
    if password is not None:
        updates["password"] = password
    return make_conninfo(dsn, **updates)


def _candidate_bin_dirs(explicit: str | Path | None = None) -> list[Path]:
    result: list[Path] = []
    if explicit:
        result.append(Path(explicit))
    for executable in ("postgres", "pg_ctl", "initdb"):
        found = shutil.which(executable)
        if found:
            result.append(Path(found).resolve().parent)
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.getenv(variable)
        if not base:
            continue
        root = Path(base) / "PostgreSQL"
        if root.is_dir():
            result.extend(path / "bin" for path in sorted(root.iterdir(), reverse=True) if path.is_dir())
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in result:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def find_native_postgres_binaries(explicit: str | Path | None = None) -> NativePostgresBinaries | None:
    suffix = ".exe" if os.name == "nt" else ""
    for bin_dir in _candidate_bin_dirs(explicit):
        postgres = bin_dir / f"postgres{suffix}"
        pg_ctl = bin_dir / f"pg_ctl{suffix}"
        initdb = bin_dir / f"initdb{suffix}"
        pg_isready = bin_dir / f"pg_isready{suffix}"
        psql = bin_dir / f"psql{suffix}"
        if postgres.is_file() and pg_ctl.is_file() and initdb.is_file() and pg_isready.is_file():
            return NativePostgresBinaries(
                bin_dir, postgres, pg_ctl, initdb, pg_isready, psql if psql.is_file() else None
            )
    return None
