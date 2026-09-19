"""Tree-sitter language registry for SecureCodeRAG."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from types import ModuleType
from typing import Any

from tree_sitter import Language

from src.models import ProgrammingLanguage


class LanguageRegistryError(RuntimeError):
    """Raised when language registry operations fail."""


class UnsupportedLanguageError(LanguageRegistryError):
    """Raised when a language has no registered Tree-sitter grammar."""


@dataclass(frozen=True)
class LanguageRegistration:
    """Registration information for a programming language."""

    language: ProgrammingLanguage
    name: str
    grammar_module: str
    grammar_function: str = "language"


class LanguageRegistry:
    """Registry that resolves programming languages to Tree-sitter grammars."""

    REGISTRATIONS: tuple[LanguageRegistration, ...] = (
        LanguageRegistration(
            language=ProgrammingLanguage.PYTHON,
            name="python",
            grammar_module="tree_sitter_python",
        ),
        LanguageRegistration(
            language=ProgrammingLanguage.JAVASCRIPT,
            name="javascript",
            grammar_module="tree_sitter_javascript",
        ),
        LanguageRegistration(
            language=ProgrammingLanguage.TYPESCRIPT,
            name="typescript",
            grammar_module="tree_sitter_typescript",
            grammar_function="language_typescript",
        ),
        LanguageRegistration(
            language=ProgrammingLanguage.JAVA,
            name="java",
            grammar_module="tree_sitter_java",
        ),
        LanguageRegistration(
            language=ProgrammingLanguage.C,
            name="c",
            grammar_module="tree_sitter_c",
        ),
        LanguageRegistration(
            language=ProgrammingLanguage.CPP,
            name="cpp",
            grammar_module="tree_sitter_cpp",
        ),
    )

    def __init__(self) -> None:
        """Initialize the language registry."""
        self._registrations = {
            registration.language: registration
            for registration in self.REGISTRATIONS
        }

    def is_supported(
        self,
        language: ProgrammingLanguage,
    ) -> bool:
        """Return whether a Tree-sitter grammar is registered."""
        if not isinstance(language, ProgrammingLanguage):
            return False

        return language in self._registrations

    def supported_languages(self) -> tuple[ProgrammingLanguage, ...]:
        """Return all registered programming languages."""
        return tuple(self._registrations)

    def get(
        self,
        language: ProgrammingLanguage,
    ) -> Language:
        """Return the Tree-sitter Language object for a language."""
        registration = self.get_registration(language)

        try:
            return self._load_language(
                registration.grammar_module,
                registration.grammar_function,
            )
        except Exception as exc:
            raise LanguageRegistryError(
                "Failed to load Tree-sitter grammar for "
                f"{language.value}"
            ) from exc

    def get_registration(
        self,
        language: ProgrammingLanguage,
    ) -> LanguageRegistration:
        """Return registration information for a language."""
        if not isinstance(language, ProgrammingLanguage):
            raise UnsupportedLanguageError(
                "language must be a ProgrammingLanguage value"
            )

        try:
            return self._registrations[language]
        except KeyError as exc:
            raise UnsupportedLanguageError(
                f"No Tree-sitter grammar registered for {language.value}"
            ) from exc

    def clear_cache(self) -> None:
        """Clear cached Tree-sitter language objects."""
        self._load_language.cache_clear()

    @staticmethod
    @cache
    def _load_language(
        grammar_module: str,
        grammar_function: str,
    ) -> Language:
        """Load and cache a Tree-sitter language grammar."""
        module = LanguageRegistry._import_grammar_module(
            grammar_module,
        )

        try:
            factory = getattr(module, grammar_function)
        except AttributeError as exc:
            raise LanguageRegistryError(
                f"Grammar module '{grammar_module}' does not expose "
                f"'{grammar_function}'"
            ) from exc

        if not callable(factory):
            raise LanguageRegistryError(
                f"Grammar attribute '{grammar_function}' in "
                f"'{grammar_module}' is not callable"
            )

        try:
            capsule: Any = factory()
            return Language(capsule)
        except Exception as exc:
            raise LanguageRegistryError(
                f"Unable to construct Tree-sitter language from "
                f"{grammar_module}.{grammar_function}()"
            ) from exc

    @staticmethod
    def _import_grammar_module(
        grammar_module: str,
    ) -> ModuleType:
        """Import a Tree-sitter grammar Python module."""
        try:
            return importlib.import_module(grammar_module)
        except ImportError as exc:
            raise LanguageRegistryError(
                f"Tree-sitter grammar package '{grammar_module}' "
                "is not installed"
            ) from exc