from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from vvb001_monitor.plant_shadow import native_postgres as native


def binaries(tmp_path: Path) -> native.NativePostgresBinaries:
    return native.NativePostgresBinaries(
        bin_dir=tmp_path,
        postgres=tmp_path / "postgres.exe",
        pg_ctl=tmp_path / "pg_ctl.exe",
        initdb=tmp_path / "initdb.exe",
        pg_isready=tmp_path / "pg_isready.exe",
        psql=tmp_path / "psql.exe",
    )


def test_server_start_returns_after_pg_ctl_launcher_instead_of_waiting_for_server(tmp_path):
    calls = []

    class Launcher:
        def wait(self, *, timeout):
            calls.append(("wait", timeout))
            return 0

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Launcher()

    native.launch_postgres(
        binaries(tmp_path),
        data_dir=tmp_path / "data",
        log_path=tmp_path / "postgres.log",
        launcher_log_path=tmp_path / "launcher.log",
        host="127.0.0.1",
        port=55432,
        popen_factory=popen,
    )
    command, kwargs = calls[0]
    assert "-W" in command
    assert kwargs["shell"] is False
    assert kwargs["stdout"] is not subprocess.PIPE
    assert calls[1] == ("wait", native.PG_CTL_LAUNCH_TIMEOUT_SECONDS)


def test_readiness_success_returns_owned_postmaster_pid(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "postmaster.pid").write_text("4242\n", encoding="utf-8")

    pid = native.wait_for_postgres(
        binaries(tmp_path),
        data_dir=data,
        log_path=tmp_path / "postgres.log",
        host="127.0.0.1",
        port=55432,
        admin_user="fixture_admin",
        run_probe=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "ready", ""),
    )
    assert pid == 4242


def test_readiness_timeout_is_bounded_and_diagnostic(tmp_path):
    values = iter((0.0, 1.0, 1.0))
    with pytest.raises(native.NativePostgresLifecycleError) as raised:
        native.wait_for_postgres(
            binaries(tmp_path),
            data_dir=tmp_path / "data",
            log_path=tmp_path / "postgres.log",
            host="127.0.0.1",
            port=55432,
            admin_user="fixture_admin",
            timeout=0.5,
            poll_seconds=0,
            run_probe=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                [], 1, "not ready", ""
            ),
            monotonic=lambda: next(values),
            sleep=lambda _seconds: None,
        )
    message = str(raised.value)
    assert "PostgreSQL readiness failed" in message
    assert "Timeout: 0.5 seconds" in message
    assert "127.0.0.1" in message and "55432" in message


def test_child_exit_before_readiness_is_detected(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "postmaster.pid").write_text("5252\n", encoding="utf-8")
    with pytest.raises(native.NativePostgresLifecycleError, match="exited before readiness"):
        native.wait_for_postgres(
            binaries(tmp_path),
            data_dir=data,
            log_path=tmp_path / "postgres.log",
            host="127.0.0.1",
            port=55432,
            admin_user="fixture_admin",
            run_probe=lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", ""),
            pid_exists=lambda _pid: False,
        )


def _patch_successful_cluster(monkeypatch, stopped: list[int]):
    monkeypatch.setattr(native, "port_is_occupied", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        native,
        "run_checked_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    monkeypatch.setattr(native, "launch_postgres", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(native, "wait_for_postgres", lambda *_args, **_kwargs: 6262)
    monkeypatch.setattr(
        native,
        "stop_owned_postgres",
        lambda *_args, **kwargs: stopped.append(kwargs["owned_pid"]),
    )


def test_cleanup_runs_after_exception(monkeypatch, tmp_path):
    stopped: list[int] = []
    _patch_successful_cluster(monkeypatch, stopped)
    with pytest.raises(RuntimeError, match="injected"):
        with native.temporary_native_postgres(
            binaries(tmp_path),
            root=tmp_path,
            host="127.0.0.1",
            port=55432,
            admin_user="fixture_admin",
            admin_password="not-printed",
        ):
            raise RuntimeError("injected")
    assert stopped == [6262]


def test_cleanup_runs_after_success(monkeypatch, tmp_path):
    stopped: list[int] = []
    _patch_successful_cluster(monkeypatch, stopped)
    with native.temporary_native_postgres(
        binaries(tmp_path),
        root=tmp_path,
        host="127.0.0.1",
        port=55432,
        admin_user="fixture_admin",
        admin_password="not-printed",
    ) as (_data, _log, owned_pid):
        assert owned_pid == 6262
    assert stopped == [6262]


def test_shutdown_fallback_targets_only_owned_pid(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "postmaster.pid").write_text("7272\n", encoding="utf-8")
    killed = []
    monkeypatch.setattr(
        native,
        "run_checked_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            native.NativePostgresLifecycleError("injected stop failure")
        ),
    )
    monkeypatch.setattr(native.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(native, "_wait_for_process_exit", lambda _pid, _timeout: True)
    monkeypatch.setattr(native, "port_is_occupied", lambda *_args, **_kwargs: False)

    native.stop_owned_postgres(
        binaries(tmp_path),
        data_dir=data,
        log_path=tmp_path / "postgres.log",
        host="127.0.0.1",
        port=55432,
        owned_pid=7272,
    )
    assert killed == [(7272, signal.SIGTERM)]


def test_occupied_port_fails_before_initialization(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "port_is_occupied", lambda *_args, **_kwargs: True)
    called = []
    monkeypatch.setattr(native, "run_checked_command", lambda *_args, **_kwargs: called.append(True))
    with pytest.raises(native.NativePostgresLifecycleError, match="already in use"):
        with native.temporary_native_postgres(
            binaries(tmp_path),
            root=tmp_path,
            host="127.0.0.1",
            port=55432,
            admin_user="fixture_admin",
            admin_password="not-printed",
        ):
            pass
    assert called == []
