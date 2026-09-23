"""Tests for the poisoning strategy registry."""

from __future__ import annotations

import pytest

from src.poisoning.base import (
    PoisoningStrategy,
)
from src.poisoning.false_api_guidance import (
    FalseApiGuidanceStrategy,
)
from src.poisoning.misleading_code import (
    MisleadingCodeStrategy,
)
from src.poisoning.registry import (
    PoisoningRegistry,
    PoisoningStrategyNotFoundError,
    PoisoningStrategyRegistrationError,
)
from src.poisoning.vulnerable_code import (
    VulnerableCodeStrategy,
)


class TestPoisoningRegistry:
    """Tests for PoisoningRegistry."""

    def test_default_categories_are_registered(self) -> None:
        registry = PoisoningRegistry()

        assert set(registry.categories) == {
            "misleading_code",
            "vulnerable_code",
            "false_api_guidance",
        }

    def test_get_misleading_code_strategy(self) -> None:
        registry = PoisoningRegistry()

        strategy = registry.get("misleading_code")

        assert isinstance(strategy, MisleadingCodeStrategy)
        assert isinstance(strategy, PoisoningStrategy)

    def test_get_vulnerable_code_strategy(self) -> None:
        registry = PoisoningRegistry()

        strategy = registry.get("vulnerable_code")

        assert isinstance(strategy, VulnerableCodeStrategy)
        assert isinstance(strategy, PoisoningStrategy)

    def test_get_false_api_guidance_strategy(self) -> None:
        registry = PoisoningRegistry()

        strategy = registry.get("false_api_guidance")

        assert isinstance(strategy, FalseApiGuidanceStrategy)
        assert isinstance(strategy, PoisoningStrategy)

    def test_get_returns_new_instance(self) -> None:
        registry = PoisoningRegistry()

        first = registry.get("misleading_code")
        second = registry.get("misleading_code")

        assert first is not second
        assert type(first) is type(second)

    def test_unknown_category_is_rejected(self) -> None:
        registry = PoisoningRegistry()

        with pytest.raises(PoisoningStrategyNotFoundError):
            registry.get("unknown_category")

    def test_empty_category_is_rejected(self) -> None:
        registry = PoisoningRegistry()

        with pytest.raises(PoisoningStrategyRegistrationError):
            registry.get("")

    def test_non_string_category_is_rejected(self) -> None:
        registry = PoisoningRegistry()

        with pytest.raises(PoisoningStrategyRegistrationError):
            registry.get(123)  # type: ignore[arg-type]

    def test_contains_operator(self) -> None:
        registry = PoisoningRegistry()

        assert "misleading_code" in registry
        assert "vulnerable_code" in registry
        assert "false_api_guidance" in registry
        assert "unknown_category" not in registry
        assert 123 not in registry

    def test_repr_contains_registered_categories(self) -> None:
        registry = PoisoningRegistry()

        representation = repr(registry)

        assert "misleading_code" in representation
        assert "vulnerable_code" in representation
        assert "false_api_guidance" in representation

    def test_empty_registry_can_be_created(self) -> None:
        registry = PoisoningRegistry(strategies={})

        assert len(registry) == 0
        assert registry.categories == ()

    def test_register_custom_strategy(self) -> None:
        registry = PoisoningRegistry(strategies={})

        registry.register(
            "misleading_code_custom",
            MisleadingCodeStrategy,
        )

        assert "misleading_code_custom" in registry

        strategy = registry.get("misleading_code_custom")

        assert isinstance(strategy, MisleadingCodeStrategy)

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = PoisoningRegistry()

        with pytest.raises(PoisoningStrategyRegistrationError):
            registry.register(
                "misleading_code",
                MisleadingCodeStrategy,
            )

    def test_duplicate_registration_can_be_overwritten(self) -> None:
        registry = PoisoningRegistry()

        registry.register(
            "misleading_code",
            VulnerableCodeStrategy,
            overwrite=True,
        )

        strategy = registry.get("misleading_code")

        assert isinstance(strategy, VulnerableCodeStrategy)

    def test_unregister_removes_strategy(self) -> None:
        registry = PoisoningRegistry()

        registry.unregister("misleading_code")

        assert "misleading_code" not in registry

        with pytest.raises(PoisoningStrategyNotFoundError):
            registry.get("misleading_code")

    def test_unregister_unknown_strategy_is_rejected(self) -> None:
        registry = PoisoningRegistry()

        with pytest.raises(PoisoningStrategyNotFoundError):
            registry.unregister("unknown_category")

    def test_clear_removes_all_strategies(self) -> None:
        registry = PoisoningRegistry()

        registry.clear()

        assert len(registry) == 0
        assert registry.categories == ()

    def test_iteration_returns_categories(self) -> None:
        registry = PoisoningRegistry()

        assert set(registry) == {
            "misleading_code",
            "vulnerable_code",
            "false_api_guidance",
        }