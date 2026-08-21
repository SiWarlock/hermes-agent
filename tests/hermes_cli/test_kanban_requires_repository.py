"""Behavior contract for tri-state Kanban repository requirements."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_repository_requirement_rejects_scratch_before_insert(kanban_home):
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

        with pytest.raises(
            ValueError,
            match="requires_repository=true is incompatible with workspace_kind=scratch",
        ):
            kb.create_task(
                conn,
                title="code task",
                workspace_kind="scratch",
                requires_repository=True,
            )

        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
        assert not any((kanban_home / "kanban" / "workspaces").glob("*"))


def test_repository_requirement_persists_true_false_and_unset(kanban_home, tmp_path):
    with kb.connect() as conn:
        true_id = kb.create_task(
            conn,
            title="repository task",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo"),
            requires_repository=True,
        )
        false_id = kb.create_task(
            conn,
            title="research task",
            workspace_kind="scratch",
            requires_repository=False,
        )
        unset_id = kb.create_task(conn, title="legacy task")

        assert kb.get_task(conn, true_id).requires_repository is True
        assert kb.get_task(conn, false_id).requires_repository is False
        assert kb.get_task(conn, unset_id).requires_repository is None
        raw = conn.execute(
            "SELECT id, requires_repository FROM tasks ORDER BY id"
        ).fetchall()
        stored = {row["id"]: row["requires_repository"] for row in raw}
        assert stored == {true_id: 1, false_id: 0, unset_id: None}


@pytest.mark.parametrize("invalid", ["true", "maybe", 1, 2, [], {}])
def test_repository_requirement_rejects_non_boolean_db_values(kanban_home, invalid):
    with kb.connect() as conn:
        with pytest.raises(ValueError, match="must be true, false, or null"):
            kb.create_task(
                conn,
                title="invalid assertion",
                workspace_kind="worktree",
                requires_repository=invalid,
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_repository_requirement_rejects_unresolved_explicit_project(kanban_home):
    with kb.connect() as conn:
        with pytest.raises(
            ValueError,
            match="requires_repository=true but project p_missing did not resolve",
        ):
            kb.create_task(
                conn,
                title="code task",
                project_id="p_missing",
                requires_repository=True,
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_project_resolution_precedes_repository_validation(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    with projects_db.connect_closing() as project_conn:
        project_id = projects_db.create_project(
            project_conn,
            name="repository contract",
            primary_path=str(repo),
        )
    if not isinstance(project_id, str):
        project_id = project_id.id

    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="code task",
            workspace_kind="scratch",
            project_id=project_id,
            requires_repository=True,
        )
        task = kb.get_task(conn, task_id)

    assert task.workspace_kind == "worktree"
    assert task.requires_repository is True


def test_dependency_child_does_not_inherit_repository_requirement(kanban_home, tmp_path):
    with kb.connect() as conn:
        parent = kb.create_task(
            conn,
            title="implementation",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo"),
            requires_repository=True,
        )
        child = kb.create_task(
            conn,
            title="review notes",
            parents=[parent],
        )

        assert kb.get_task(conn, child).requires_repository is None


def test_task_from_row_tolerates_missing_repository_column(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy reader")
        row = conn.execute(
            "SELECT id, title, body, assignee, status, priority, created_by, "
            "created_at, started_at, completed_at, workspace_kind, workspace_path, "
            "claim_lock, claim_expires, tenant FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()

    assert kb.Task.from_row(row).requires_repository is None


def test_legacy_db_migrates_repository_requirement_as_null(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    legacy.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
            status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER,
            completed_at INTEGER, workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT, claim_lock TEXT, claim_expires INTEGER
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL
        );
        INSERT INTO tasks (id, title, status, created_at)
        VALUES ('legacy', 'old task', 'ready', 1);
        """
    )
    legacy.commit()
    legacy.close()

    kb.init_db(db_path=db_path)
    with kb.connect(db_path) as conn:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(tasks)")
        }
        task = kb.get_task(conn, "legacy")

    assert "requires_repository" in columns
    assert task.requires_repository is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, "yes"), (False, "no")],
)
def test_worker_context_exposes_repository_requirement(
    kanban_home, tmp_path, value, expected
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="worker contract",
            workspace_kind="worktree" if value else "scratch",
            workspace_path=str(tmp_path / "repo") if value else None,
            requires_repository=value,
        )
        context = kb.build_worker_context(conn, task_id)

    assert f"Requires repository: {expected}" in context


def test_worker_context_omits_unset_repository_requirement(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy contract")
        context = kb.build_worker_context(conn, task_id)

    assert "Requires repository:" not in context


def _repository_triage_root(conn, tmp_path):
    return kb.create_task(
        conn,
        title="repository epic",
        triage=True,
        workspace_kind="worktree",
        workspace_path=str(tmp_path / "repo"),
        requires_repository=True,
    )


def test_decomposition_child_inherits_repository_requirement(kanban_home, tmp_path):
    with kb.connect() as conn:
        root_id = _repository_triage_root(conn, tmp_path)
        child_ids = kb.decompose_triage_task(
            conn,
            root_id,
            root_assignee="orchestrator",
            children=[{"title": "implementation"}],
        )

        assert child_ids is not None
        assert kb.get_task(conn, child_ids[0]).requires_repository is True


def test_decomposition_cannot_downgrade_repository_requirement_atomically(
    kanban_home, tmp_path
):
    with kb.connect() as conn:
        root_id = _repository_triage_root(conn, tmp_path)
        with pytest.raises(
            ValueError,
            match=(
                r"child\[0\] sets requires_repository=false but root task "
                rf"{root_id} requires a repository"
            ),
        ):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=[
                    {"title": "implementation", "requires_repository": False}
                ],
            )

        assert kb.get_task(conn, root_id).status == "triage"
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE id != ?", (root_id,)
        ).fetchone()[0] == 0


def test_decomposition_repository_child_cannot_override_to_scratch(
    kanban_home, tmp_path
):
    with kb.connect() as conn:
        root_id = _repository_triage_root(conn, tmp_path)
        with pytest.raises(
            ValueError,
            match="requires_repository=true is incompatible with workspace_kind=scratch",
        ):
            kb.decompose_triage_task(
                conn,
                root_id,
                root_assignee="orchestrator",
                children=[
                    {"title": "implementation", "workspace_kind": "scratch"}
                ],
            )


def test_repository_required_dir_must_resolve_inside_git_repository(
    kanban_home, tmp_path
):
    plain_dir = tmp_path / "plain"
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="implementation",
            workspace_kind="dir",
            workspace_path=str(plain_dir),
            requires_repository=True,
        )
        task = kb.get_task(conn, task_id)

    with pytest.raises(ValueError, match="not inside a git repository"):
        kb.resolve_workspace(task)
    assert plain_dir.is_dir()
