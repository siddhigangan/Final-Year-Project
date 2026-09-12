"""
Foundation unit tests for SecureCodeRAG.
Verifies environment setup, package imports, configuration parsing, and logging functionality.
"""

import importlib
import logging
import os
import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, load_config
from src.logger import get_logger


class TestFoundation(unittest.TestCase):
    """Test cases to validate Step 1 project foundation."""

    def test_package_imports(self):
        """Verify that all core modular packages can be imported cleanly."""
        modules = [
            "src",
            "src.config",
            "src.logger",
            "src.ingestion",
            "src.chunking",
            "src.embeddings",
            "src.vectorstore",
            "src.retrieval",
            "src.generation",
            "src.poisoning",
            "src.defense",
            "src.evaluation",
        ]
        for mod_name in modules:
            with self.subTest(module=mod_name):
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod)

    def test_default_config_loading(self):
        """Verify loading default configuration from JSON file."""
        config = load_config()
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.project_name, "SecureCodeRAG")
        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.top_k, 5)

    def test_config_env_override(self):
        """Verify environment variables override configuration defaults."""
        os.environ["CHUNK_SIZE"] = "1024"
        os.environ["LOG_LEVEL"] = "DEBUG"
        try:
            config = load_config()
            self.assertEqual(config.chunk_size, 1024)
            self.assertEqual(config.log_level, "DEBUG")
        finally:
            os.environ.pop("CHUNK_SIZE", None)
            os.environ.pop("LOG_LEVEL", None)

    def test_logging_initialization(self):
        """Verify logger creation and file/console logging capability."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = os.path.join(tmp_dir, "test.log")
            logger = get_logger(name="TestLogger", log_file=log_file, level="DEBUG")

            self.assertEqual(logger.name, "TestLogger")
            self.assertEqual(logger.level, logging.DEBUG)

            logger.info("Test log entry")

            # Force handler flush and close to allow file cleanup on Windows
            for handler in logger.handlers[:]:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)

            self.assertTrue(os.path.exists(log_file))
            with open(log_file, "r", encoding="utf-8") as f:
                content = f.read()
                self.assertIn("Test log entry", content)
                self.assertIn("TestLogger", content)

    def test_directory_structure_exists(self):
        """Verify essential project directory structure exists."""
        project_root = Path(__file__).resolve().parent.parent
        expected_dirs = [
            "src",
            "src/ingestion",
            "src/chunking",
            "src/embeddings",
            "src/vectorstore",
            "src/retrieval",
            "src/generation",
            "src/poisoning",
            "src/defense",
            "src/evaluation",
            "data/clean",
            "data/poisoned",
            "data/processed",
            "configs",
            "tests",
            "experiments",
            "results",
            "notebooks",
            "docs",
        ]
        for relative_dir in expected_dirs:
            dir_path = project_root / relative_dir
            with self.subTest(directory=relative_dir):
                self.assertTrue(dir_path.exists(), f"Directory missing: {relative_dir}")
                self.assertTrue(dir_path.is_dir(), f"Path is not directory: {relative_dir}")


if __name__ == "__main__":
    unittest.main()
