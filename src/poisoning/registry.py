"""Registry and factory for poisoning strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from src.poisoning.base import (
    PoisoningConfigurationError,
    PoisoningStrategy,
)
from src.poisoning.false_api_guidance import FalseApiGuidanceStrategy
from src.poisoning.misleading_code import MisleadingCodeStrategy
from src.poisoning.vulnerable_code import VulnerableCodeStrategy


class PoisoningRegistryError(Exception):
    """Base exception for poisoning registry errors."""


class PoisoningStrategyNotFoundError(PoisoningRegistryError):
    """Raised when a requested poisoning strategy is not registered."""


class PoisoningStrategyRegistrationError(PoisoningRegistryError):
    """Raised when strategy registration is invalid."""


StrategyClass: TypeAlias = type[PoisoningStrategy]


DEFAULT_STRATEGIES: dict[str, StrategyClass] = {
    "misleading_code": MisleadingCodeStrategy,
    "vulnerable_code": VulnerableCodeStrategy,
    "false_api_guidance": FalseApiGuidanceStrategy,
}


@dataclass
class PoisoningRegistry:
    """Registry used to create poisoning strategy instances.

    The registry maps stable poisoning category names to strategy
    implementations. A fresh strategy instance is created for every
    call to :meth:`get`.
    """

    strategies: dict[str, StrategyClass] | None = None
    _strategies: dict[str, StrategyClass] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Initialize and validate the registry."""
        configured = (
            DEFAULT_STRATEGIES
            if self.strategies is None
            else self.strategies
        )

        self._strategies = {}

        for category, strategy_class in configured.items():
            self.register(category, strategy_class)

    @property
    def categories(self) -> tuple[str, ...]:
        """Return registered categories in deterministic order."""
        return tuple(self._strategies)

    def register(
        self,
        category: str,
        strategy_class: StrategyClass,
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a poisoning strategy class.

        Args:
            category: Stable poisoning category identifier.
            strategy_class: Concrete poisoning strategy class.
            overwrite: Replace an existing registration when True.

        Raises:
            PoisoningStrategyRegistrationError:
                If the category or strategy class is invalid, or if the
                category is already registered.
        """
        self._validate_category(category)

        if not isinstance(strategy_class, type):
            raise PoisoningStrategyRegistrationError(
                "strategy_class must be a class."
            )

        if not issubclass(strategy_class, PoisoningStrategy):
            raise PoisoningStrategyRegistrationError(
                "strategy_class must inherit from PoisoningStrategy."
            )

        if category in self._strategies and not overwrite:
            raise PoisoningStrategyRegistrationError(
                f"Poisoning strategy {category!r} is already registered."
            )

        self._strategies[category] = strategy_class

    def unregister(self, category: str) -> None:
        """Remove a registered poisoning strategy.

        Raises:
            PoisoningStrategyNotFoundError:
                If the category is not registered.
        """
        self._validate_category(category)

        if category not in self._strategies:
            raise PoisoningStrategyNotFoundError(
                f"No poisoning strategy registered for {category!r}."
            )

        del self._strategies[category]

    def get(
        self,
        category: str,
        *,
        seed: int = 42,
        parameters: dict | None = None,
    ) -> PoisoningStrategy:
        """Create and return a strategy for the requested category.

        Args:
            category: Registered poisoning category.
            seed: Deterministic seed passed to the strategy.
            parameters: Optional strategy configuration.

        Returns:
            A newly created poisoning strategy instance.

        Raises:
            PoisoningStrategyNotFoundError:
                If the category is unknown.
            PoisoningConfigurationError:
                If the requested configuration is invalid.
        """
        self._validate_category(category)

        strategy_class = self._strategies.get(category)

        if strategy_class is None:
            raise PoisoningStrategyNotFoundError(
                f"No poisoning strategy registered for {category!r}."
            )

        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PoisoningConfigurationError(
                "seed must be an integer."
            )

        if parameters is not None and not isinstance(parameters, dict):
            raise PoisoningConfigurationError(
                "parameters must be a dictionary or None."
            )

        # The strategy implementations currently receive configuration
        # through PoisoningInput rather than their constructors.
        #
        # We therefore construct the strategy first and attach the
        # registry-level configuration only when the implementation
        # explicitly supports those attributes.
        strategy = strategy_class()

        if hasattr(strategy, "seed"):
            try:
                strategy.seed = seed
            except (AttributeError, TypeError) as exc:
                raise PoisoningConfigurationError(
                    f"Unable to configure seed for {category!r}."
                ) from exc

        if hasattr(strategy, "parameters"):
            try:
                strategy.parameters = {} if parameters is None else dict(parameters)
            except (AttributeError, TypeError) as exc:
                raise PoisoningConfigurationError(
                    f"Unable to configure parameters for {category!r}."
                ) from exc

        return strategy

    def contains(self, category: str) -> bool:
        """Return whether a category is registered."""
        self._validate_category(category)
        return category in self._strategies

    def __contains__(self, category: object) -> bool:
        """Support ``category in registry``."""
        if not isinstance(category, str):
            return False

        return category in self._strategies

    def __len__(self) -> int:
        """Return the number of registered strategies."""
        return len(self._strategies)

    def __iter__(self):
        """Iterate over registered category names."""
        return iter(self._strategies)

    def __repr__(self) -> str:
        """Return a concise registry representation."""
        categories = ", ".join(self._strategies)
        return f"PoisoningRegistry(categories=[{categories}])"

    def clear(self) -> None:
        """Remove all registered strategies."""
        self._strategies.clear()

    @staticmethod
    def _validate_category(category: str) -> None:
        """Validate a poisoning category."""
        if not isinstance(category, str):
            raise PoisoningStrategyRegistrationError(
                "category must be a string."
            )

        if not category.strip():
            raise PoisoningStrategyRegistrationError(
                "category must not be empty."
            )

        if category != category.strip():
            raise PoisoningStrategyRegistrationError(
                "category must not contain leading or trailing whitespace."
            )