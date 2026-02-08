import signal
import threading
import time

import uvicorn

from app.config import settings
from app.gradio_ui import build_gradio_app, launch_gradio_app
from app.utils import ensure_directories


def main() -> None:
    ensure_directories()

    stop_event = threading.Event()
    gradio_holder: dict[str, object] = {}

    config = uvicorn.Config(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    api_server = uvicorn.Server(config)

    def _handle_signal(signum: int, _frame: object) -> None:
        print(f"Received signal {signum}, shutting down.")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    api_thread = threading.Thread(target=api_server.run, name="uvicorn-server", daemon=True)
    api_thread.start()

    demo = build_gradio_app()
    launch_gradio_app(demo=demo, prevent_thread_lock=True)
    gradio_holder["demo"] = demo

    print(f"FastAPI: http://{settings.api_host}:{settings.api_port}")
    print(f"Swagger: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"Gradio:  http://{settings.gradio_host}:{settings.gradio_port}")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
            if not api_thread.is_alive():
                stop_event.set()
    finally:
        api_server.should_exit = True
        demo_obj = gradio_holder.get("demo")
        if demo_obj is not None and hasattr(demo_obj, "close"):
            demo_obj.close()  # type: ignore[attr-defined]
        api_thread.join(timeout=20)


if __name__ == "__main__":
    main()
