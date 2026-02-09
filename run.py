import logging
import signal
import threading
import time

from dotenv import load_dotenv

import uvicorn

from app.core.logging import configure_logging


def main() -> None:
    load_dotenv(override=True)
    log_file = configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting HunyuanVideo service. log_file=%s", log_file)

    from app.core.config import settings
    from app.ui.gradio_app import build_gradio_app, launch_gradio_app
    from app.utils.common import ensure_directories

    ensure_directories()

    stop_event = threading.Event()
    gradio_holder: dict[str, object] = {}

    config = uvicorn.Config(
        "app.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        log_config=None,
    )
    api_server = uvicorn.Server(config)

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %s, shutting down.", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)

    api_thread = threading.Thread(target=api_server.run, name="uvicorn-server", daemon=True)
    api_thread.start()

    demo = build_gradio_app()
    launch_gradio_app(demo=demo, prevent_thread_lock=True)
    gradio_holder["demo"] = demo

    logger.info("FastAPI: http://%s:%s", settings.api_host, settings.api_port)
    logger.info("Swagger: http://%s:%s/docs", settings.api_host, settings.api_port)
    logger.info("Gradio: http://%s:%s", settings.gradio_host, settings.gradio_port)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
            if not api_thread.is_alive():
                stop_event.set()
    finally:
        logger.info("Stopping API/Gradio services.")
        api_server.should_exit = True
        demo_obj = gradio_holder.get("demo")
        if demo_obj is not None and hasattr(demo_obj, "close"):
            demo_obj.close()  # type: ignore[attr-defined]
        api_thread.join(timeout=20)
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
