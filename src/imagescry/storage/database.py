"""Database connection and session management for ImageScry application."""

from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Self

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine, select

from imagescry.storage.base import BaseStorageModel, StorageModel


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

    def add_item(self, item: BaseStorageModel) -> int:
        """Add a single item to the database.

        Args:
            item (BaseStorageModel): Item to add to the database.

        Returns:
            int: ID of the added item.
        """
        return self.add_items([item])[0]

    def add_items(self, items: list[StorageModel]) -> list[int]:
        """Add multiple items to the database.

        Args:
            items (list[StorageModel]): List of items to add to the database.

        Returns:
            list[int]: List of IDs of added items.

        Raises:
            RuntimeError: If the database engine is not initialized or if adding items fails.

        """
        # Quick exit if no items to add
        if not items:
            return []

        # Check if the engine is initialized
        if not self.engine:
            raise RuntimeError("Database engine is not initialized. Call create_engine() first.")

        with Session(self.engine) as session:
            try:
                # Add and commit the items to the session
                session.add_all(items)
                session.commit()

                # Get the IDs of the added items
                ids: list[int] = []
                for item in items:
                    if item.id is not None:
                        ids.append(item.id)
                    else:
                        raise RuntimeError("Item ID is None after commit, indicating a failure to add to the database.")

                return ids

            except Exception as e:
                # Rollback the session in case of an error
                session.rollback()
                raise RuntimeError(f"Failed to add items to the database: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None

    def create_engine(
        self, *, check_same_thread: bool = False, timeout: float = 30.0, pool_recycle: int = 3_600
    ) -> Engine:
        """Create a new database engine.

        Args:
            check_same_thread (bool): If True, connection object created in one thread cannot be used in another thread.
                Default is False, which allows connections to be shared across threads.
            timeout (float): Timeout for database connections in seconds. Default is 30.0.
            pool_recycle (int): Time in seconds to recycle connections in the pool. Default is 3600 (1 hour).

        Returns:
            Engine: The SQLAlchemy engine for the SQLite database.
        """
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

    def get_item(self, model: type[StorageModel], item_id: int) -> StorageModel | None:
        """Get a single item from the database by ID.

        Args:
            model (type[StorageModel]): The model class to query.
            item_id (int): ID of the item to retrieve.

        Returns:
            StorageModel | None: The retrieved item or None if not found.
        """
        with Session(self.engine) as session:
            return session.get(model, item_id)

    def get_items(self, model: type[StorageModel], item_ids: list[int] | None = None) -> list[StorageModel]:
        """Get multiple items from the database by their IDs.

        Args:
            model (type[StorageModel]): The model class to query.
            item_ids (list[int] | None): List of IDs of the items to retrieve or None to retrieve all items.
                Default is None.


        Returns:
            list[StorageModel]: The retrieved items or an empty list if none found.
        """
        with Session(self.engine) as session:
            # Define selection statement based on whether item_ids is provided
            if item_ids is None:
                statement = select(model)
            else:
                # If IDs are provided, filter by those IDs
                if not item_ids:
                    return []
                statement = select(model).where(model.id.in_(item_ids))

            return list(session.exec(statement).all())

    def get_item_ids(self, model: type[StorageModel]) -> list[int]:
        """Get all item IDs for a given model.

        Args:
            model (type[StorageModel]): The model class (table) to query.

        Returns:
            list[int]: List of IDs of all (non-null) items in the specified table.
        """
        with Session(self.engine) as session:
            statement = select(model.id)
            return [obj_id for obj_id in session.exec(statement) if obj_id is not None]

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
