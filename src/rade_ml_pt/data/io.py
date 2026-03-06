import os
import json
import pickle
import logging
import pandas as pd
from typing import Union, Any, Callable, Dict, List

from src.rade_ml_pt.validation.exceptions import CacheLoaderError, FileLoadError, FileSaveError

# define module logger file.
logger = logging.getLogger(__name__)


class CacheLoader:
    """
    A class that caches data in memory and loads from file if not already cached.

    Supports .pkl, .json, .csv & .parquet formats.

    Note: ``cache`` is a **class-level** mutable dict shared across all call sites.
    This is intentional singleton state so that data loaded once is reused everywhere
    in the same process. Call ``cache.clear()`` between independent runs if isolation
    is required.
    """
    cache: Dict[str, Any] = {}

    # ------ File Loaders ------ #
    @classmethod
    def _load_pickle(cls, file_path: str) -> Any:
        """Loads and returns data from pickle file."""
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            raise FileLoadError(f"Pickle file error: {e}")

    @classmethod
    def _load_json(cls, file_path: str) -> Any:
        """Loads and returns data from json file."""
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                return json.load(f)
        except (json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            raise FileLoadError(f"Json decoding error: {e}")

    @classmethod
    def _load_csv(cls, file_path: str) -> Any:
        """Loads and returns data from csv as a list of dictionaries."""
        try:
            return pd.read_csv(file_path, index_col=0)
        except pd.errors.ParserError as e:
            raise FileLoadError(f"CSV parsing error: {e}")

    @classmethod
    def _load_parquet(cls, file_path: str) -> Any:
        """Loads and returns data from parquet file."""
        try:
            return pd.read_parquet(file_path)
        except ValueError as e:
            raise FileLoadError(f"Parquet rea error for, {file_path}: {e}") from e
        except ImportError as e:
            raise FileLoadError(f"Missing parquet engine for, {file_path}: {e}") from e
        except OSError as e:
            raise FileLoadError(f"Error accessing parquet file {file_path}: {e}") from e

    # ------ File Writers ------ #
    @classmethod
    def _save_pickle(cls, data: Any, file_path: str) -> None:
        """Saves data to pickle file."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        except (pickle.PicklingError, TypeError) as e:
            raise FileSaveError(f"Pickling error for file: {file_path}: {e}") from e
        except OSError as e:
            raise FileSaveError(f"Failed to write pickle file, {file_path}: {e}") from e

    @classmethod
    def _save_json(cls, data: Any, file_path: str) -> None:
        """Saves data to json file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except (TypeError, OverflowError) as e:
            raise FileSaveError(f"Json serialisation error for file: {file_path}: {e}") from e
        except OSError as e:
            raise FileSaveError(f"Failed to write json file, {file_path}: {e}") from e

    @classmethod
    def _save_csv(cls, data: Union[pd.DataFrame, List[Dict[str, Any]]], file_path: str) -> None:
        """Saves data to csv file."""
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError("CSV data must be in a list of dictionaries or pandas dataframe")
        except ValueError as e:
            raise FileSaveError(f"CSV serialization for file: {file_path}: {e}")
        except OSError as e:
            raise FileSaveError(f"Failed to write csv file, {file_path}: {e}") from e

    @classmethod
    def _save_parquet(cls, data: Any, file_path: str) -> None:
        """Saves data to parquet file."""
        try:
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            df.to_parquet(file_path, index=False)
        except ValueError as e:
            raise FileSaveError(f"Parquet serialization for file: {file_path}: {e}")
        except OSError as e:
            raise FileSaveError(f"Failed to write parquet file, {file_path}: {e}")

    # ------ Main API methods ------ #
    @classmethod
    def _get_loader(cls, extension: str) -> Union[Callable, None]:
        """Get correct data loader for file extension."""
        return {
            '.pkl': cls._load_pickle, '.json': cls._load_json, '.csv': cls._load_csv, '.parquet': cls._load_parquet,
        }.get(extension)

    @classmethod
    def _get_saver(cls, extension: str) -> Union[Callable, None]:
        """Get correct data saver for file extension."""
        return {
            '.pkl': cls._save_pickle, '.json': cls._save_json, '.csv': cls._save_csv, '.parquet': cls._save_parquet,
        }.get(extension)

    @classmethod
    def _get_extension(cls, path: str) -> str:
        """Extracts file extension from path."""
        return os.path.splitext(path)[1].lower()

    @classmethod
    def get(cls, key: str, file_path: str) -> Any:
        """Retrieves data from memory if cached, otherwise loads from file and caches it."""
        if key in cls.cache:
            logger.info(f"Extracting {key} from cache.")
            return cls.cache[key]

        extension = cls._get_extension(path=file_path)
        loader = cls._get_loader(extension=extension)
        if not loader:
            raise FileLoadError(f"Unsupported file extension: {extension}")
        data = loader(file_path)
        cls.cache[key] = data
        logger.info(f"Extracting {key} from file, {file_path}.")
        return data

    @classmethod
    def save(cls, key: str, file_path: str) -> None:
        """Saves cached data to file, file type determined by extension."""
        if key not in cls.cache:
            raise FileSaveError(f"No cached data found for key {key}.")
        extension = cls._get_extension(file_path)
        saver = cls._get_saver(extension)
        if not saver:
            raise FileSaveError(f"Unsupported file extension for save: {extension}")
        saver(cls.cache[key], file_path)

    @classmethod
    def save_data(cls, data: Any, file_path: str) -> None:
        """Saves un-cached data to file, file type determined by extension."""
        extension = cls._get_extension(path=file_path)
        saver = cls._get_saver(extension)
        if not saver:
            raise FileSaveError(f"Unsupported file extension for save: {extension}")
        saver(data, file_path)

    @classmethod
    def load_save_data(cls, key: str, data: Any, file_path: str) -> None:
        """"Load data into cache, then save data to file path."""
        # update cache with data.
        cls.update_cache(key, data)

        # save data to file.
        cls.save_data(data, file_path)

    @classmethod
    def update_cache(cls, key: str, data: Any, return_data: bool = False) -> Union[None, Any]:
        """Updates the cache with data under the specified key."""
        cls.cache[key] = data
        logger.info(f"{key} updated in cache.")
        if return_data:
            return cls.cache[key]
        return None

    @classmethod
    def update_cache_dict(cls, key: str, data: Any) -> None:
        """Update the dictionary in cache with data under specific key."""
        # check whether data key in cache is a dictionary.
        if not isinstance(cls.cache[key], dict):
            raise CacheLoaderError(f"Cached data, {key} is not of type dictionary.")
        cls.cache[key].update(data)
        logger.info(f"{key} updated in cache.")

    @classmethod
    def load(cls, file_path: str) -> Any:
        """Load data from file path and return it."""
        extension = cls._get_extension(path=file_path)
        loader = cls._get_loader(extension=extension)
        if not loader:
            raise FileLoadError(f"Unsupported file extension for load: {extension}")
        data = loader(file_path)
        logger.info(f"Data loaded into cache from file {file_path}.")
        return data

    @classmethod
    def reload(cls, key: str, file_path: str) -> Any:
        """Forces reload of data from file into cache and returns it."""
        extension = cls._get_extension(path=file_path)
        loader = cls._get_loader(extension=extension)
        if not loader:
            raise FileLoadError(f"Unsupported file extension for load: {extension}")
        data = loader(file_path)
        cls.cache[key] = data
        logger.info(f"{key} reloaded into cache from file {file_path}.")
        return data

    @classmethod
    def get_cached(cls, key: str) -> Any:
        """Returns data from cache is available."""
        logger.info(f"Extracting {key} from cache.")
        return cls.cache.get(key)
