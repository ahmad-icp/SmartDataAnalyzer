"""Background task helper using ThreadPoolExecutor.

Provides a simple way to run long-running functions off the Streamlit main thread
and poll for completion via session state. Uses a cached executor to reuse threads.

Design: minimal, safe wrapper. Avoids direct Streamlit calls from background threads.
"""
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import uuid
from typing import Callable, Any


def _get_executor(max_workers: int = 2) -> ThreadPoolExecutor:
    # cache executor as a Streamlit resource to persist across reruns
    if "_bg_executor" not in st.session_state:
        st.session_state["_bg_executor"] = ThreadPoolExecutor(max_workers=max_workers)
    return st.session_state["_bg_executor"]


def run_in_background(fn: Callable[..., Any], *args, task_name: str | None = None, **kwargs) -> str:
    """Submit `fn(*args, **kwargs)` to background executor and return task id.

    The Future is stored in `st.session_state['bg_tasks'][task_id]`.
    """
    execu = _get_executor()
    if "bg_tasks" not in st.session_state:
        st.session_state["bg_tasks"] = {}
    task_id = task_name or str(uuid.uuid4())
    future = execu.submit(fn, *args, **kwargs)
    st.session_state["bg_tasks"][task_id] = future
    return task_id


def get_task_status(task_id: str):
    tasks = st.session_state.get("bg_tasks", {})
    fut = tasks.get(task_id)
    if fut is None:
        return "not_found", None
    if fut.running():
        return "running", None
    if fut.cancelled():
        return "cancelled", None
    if fut.done():
        try:
            res = fut.result()
            return "done", res
        except Exception as e:
            return "error", e
    return "unknown", None
