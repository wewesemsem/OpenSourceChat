"""Start the server on the address from config.yaml."""

import uvicorn

from app.main import app, config


def main() -> None:
    # A local app should not trust X-Forwarded-* from the client.
    # The bind-address warning is logged when the app starts.
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        proxy_headers=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
