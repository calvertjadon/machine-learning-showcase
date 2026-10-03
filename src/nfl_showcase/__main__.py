"""Support ``python -m nfl_showcase`` by delegating to the console entry point."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
