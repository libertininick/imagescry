"""Database connection and session management for ImageScry application."""

from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Self

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine


class DatabaseManager:
    """Manager for an ImageScry SQLite database stored in a local directory.

    This class handles the creation, deletion, and session management of the SQLite database
    used by the ImageScry application per image directory.

    Attributes:
        database_name (ClassVar[str]): The name of the SQLite database file. Default is "imagescry.db".

    Examples:
        ```python
        from imagescry.storage import DatabaseManager

        # Initialize the database manager for a specific directory
        db_manager = DatabaseManager("/path/to/image/directory")

        # Use a context manager to ensure proper resource management
        with db_manager as db:
            # Get a session to interact with the database
            with db.get_session() as session:
                # Perform database operations here
                pass
        ```
    """

    database_name: ClassVar[str] = "imagescry.db"

    def __init__(self, db_dir: str | PathLike) -> None:
        """Initialize the DatabaseManager.

        Args:
            db_dir (str | PathLike): Directory to store the SQLite database file, or where it already exists.

        Raises:
            ValueError: If the provided directory does not exist.
        """
        self.db_dir = Path(db_dir)
        if not self.db_dir.exists():
            raise ValueError(f"Directory {self.db_dir} does not exist.")

        # Define database path and URL
        self.db_path = self.db_dir / self.database_name
        self.db_url = f"sqlite:///{self.db_path}"

        # Create the database engine
        self.engine: Engine | None = self.create_engine()

        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.engine)

    def __del__(self) -> None:
        """Destructor to ensure the database connection is closed."""
        self.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Context manager exit - close the database connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def create_engine(
        self, *, check_same_thread: bool = False, timeout: float = 30.0, pool_recycle: int = 3_600
    ) -> Engine:
        """Create a new database engine."""
        return create_engine(
            self.db_url,
            echo=False,
            connect_args={
                "check_same_thread": check_same_thread,
                "timeout": timeout,
            },
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=pool_recycle,  # Connection recycle time (in seconds)
        )

    def delete_database(self) -> bool:
        """Delete database.

        Returns:
            bool: True if database was deleted, False if it did not exist.
        """
        self.close()

        if self.db_path.exists():
            self.db_path.unlink()
            return True

        return False

    def get_session(self) -> Session:
        """Get a database session."""
        if not self.engine:
            raise RuntimeError("Database engine is not initialized. Call create_engine() first.")

        return Session(self.engine)

    @property
    def is_connected(self) -> bool:
        """bool: Check if the database engine is active."""
        return self.engine is not None

    @property
    def database_exists(self) -> bool:
        """bool: Check if the database file exists."""
        return self.db_path.exists()
