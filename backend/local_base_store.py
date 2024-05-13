import os
import re
import json
from pathlib import Path
from typing import Iterator, List, Optional, Generic, Sequence, Tuple, TypeVar, Union

from langchain_core.stores import BaseStore
from langchain.storage.exceptions import InvalidKeyException
from langchain_core.documents import Document



class LocalBaseStore(BaseStore[str, Document]):
    """BaseStore interface that works on the local file system.

    Examples:
        Create a LocalBaseStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import LocalBaseStore

            # Instantiate the LocalBaseStore with the root path
            base_store = LocalBaseStore("/path/to/root")

            # Set values for keys
            base_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = base_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            base_store.mdelete(["key1"])

            # Iterate over keys
            for key in base_store.yield_keys():
                print(key)  # noqa: T201

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir

    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise InvalidKeyException(f"Invalid characters in key: {key}")
        full_path = os.path.abspath(self.root_path / key)
        common_path = os.path.commonpath([str(self.root_path), full_path])
        if common_path != str(self.root_path):
            raise InvalidKeyException(
                f"Invalid key: {key}. Key should be relative to the full path."
                f"{self.root_path} vs. {common_path} and full path of {full_path}"
            )

        return Path(full_path)

    def _get_metadata_path(self, key: str) -> Path:
        return self._get_full_path(f"{key}_md")

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                os.chmod(dir, self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        values: List[Optional[Document]] = []
        for key in keys:
            metadata = {}
            page_content = None
            full_path = self._get_full_path(key)
            metadata_path = self._get_metadata_path(key)
            if full_path.exists():
                page_content = full_path.read_text()
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
            if page_content:
                values.append(Document(page_content, metadata=metadata))
        return values

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, document in key_value_pairs:
            full_path = self._get_full_path(key)
            metadata_path = self._get_metadata_path(key)
            self._mkdir_for_store(full_path.parent)
            self._mkdir_for_store(metadata_path.parent)
            full_path.write_text(document.page_content)
            metadata_path.write_text(json.dumps(document.metadata))
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)
                os.chmod(metadata_path, self.chmod_file)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()
            metadata_path = self._get_metadata_path(key)
            if metadata_path.exists():
                metadata_path.unlink()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix_path = self._get_full_path(prefix) if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)
