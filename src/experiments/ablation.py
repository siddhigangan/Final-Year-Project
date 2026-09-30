from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class AblationError(Exception):
    """Base exception for ablation configuration errors."""


class AblationConfigurationError(AblationError, ValueError):
    """Raised when an ablation configuration is invalid."""


class DefenseLayer(str, Enum):
    """Canonical defense layers used by the project."""

    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


@dataclass(frozen=True)
class AblationConfiguration:
    """
    Configuration for one defense ablation experiment.

    Each configuration represents the defense layers enabled for
    a particular ablation run.
    """

    name: str
    layers: tuple[DefenseLayer, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AblationConfigurationError(
                "Ablation configuration name must be a non-empty string."
            )

        if not isinstance(self.layers, tuple):
            raise AblationConfigurationError(
                "Ablation layers must be provided as a tuple."
            )

        if any(
            not isinstance(layer, DefenseLayer)
            for layer in self.layers
        ):
            raise AblationConfigurationError(
                "Ablation layers must contain only DefenseLayer values."
            )

        if len(set(self.layers)) != len(self.layers):
            raise AblationConfigurationError(
                "Ablation layers must not contain duplicates."
            )

        expected_order = tuple(
            DefenseLayer(
                f"L{index}"
            )
            for index in range(1, len(self.layers) + 1)
        )

        if self.layers != expected_order:
            raise AblationConfigurationError(
                "Ablation layers must form a sequential prefix "
                "starting from L1."
            )

    @property
    def enabled_layers(self) -> tuple[str, ...]:
        """Return enabled layer names."""

        return tuple(layer.value for layer in self.layers)

    @property
    def layer_count(self) -> int:
        """Return the number of enabled defense layers."""

        return len(self.layers)

    @property
    def is_baseline(self) -> bool:
        """Return True when no defense layer is enabled."""

        return not self.layers

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "layers": list(self.enabled_layers),
            "layer_count": self.layer_count,
            "is_baseline": self.is_baseline,
        }


ABLATION_STAGES: tuple[AblationConfiguration, ...] = (
    AblationConfiguration(
        name="no_defense",
        layers=(),
    ),
    AblationConfiguration(
        name="L1",
        layers=(DefenseLayer.L1,),
    ),
    AblationConfiguration(
        name="L1_L2",
        layers=(
            DefenseLayer.L1,
            DefenseLayer.L2,
        ),
    ),
    AblationConfiguration(
        name="L1_L2_L3",
        layers=(
            DefenseLayer.L1,
            DefenseLayer.L2,
            DefenseLayer.L3,
        ),
    ),
    AblationConfiguration(
        name="L1_L2_L3_L4",
        layers=(
            DefenseLayer.L1,
            DefenseLayer.L2,
            DefenseLayer.L3,
            DefenseLayer.L4,
        ),
    ),
    AblationConfiguration(
        name="L1_L2_L3_L4_L5",
        layers=(
            DefenseLayer.L1,
            DefenseLayer.L2,
            DefenseLayer.L3,
            DefenseLayer.L4,
            DefenseLayer.L5,
        ),
    ),
)


def get_ablation_stages() -> tuple[AblationConfiguration, ...]:
    """Return the complete ordered ablation progression."""

    return ABLATION_STAGES


def get_ablation_stage(name: str) -> AblationConfiguration:
    """Return an ablation configuration by name."""

    if not isinstance(name, str) or not name.strip():
        raise AblationConfigurationError(
            "Ablation stage name must be a non-empty string."
        )

    for stage in ABLATION_STAGES:
        if stage.name == name:
            return stage

    raise AblationConfigurationError(
        f"Unknown ablation stage: {name!r}."
    )


def build_ablation_configuration(
    layers: Iterable[DefenseLayer | str],
) -> AblationConfiguration:
    """
    Build one ablation configuration from defense layers.

    The layers must form a sequential prefix:

        ()
        (L1,)
        (L1, L2)
        ...
        (L1, L2, L3, L4, L5)
    """

    normalized_layers: list[DefenseLayer] = []

    for layer in layers:
        if isinstance(layer, DefenseLayer):
            normalized_layers.append(layer)
            continue

        if isinstance(layer, str):
            try:
                normalized_layers.append(DefenseLayer(layer))
            except ValueError as exc:
                raise AblationConfigurationError(
                    f"Unknown defense layer: {layer!r}."
                ) from exc
            continue

        raise AblationConfigurationError(
            "Defense layers must be DefenseLayer values or strings."
        )

    layer_tuple = tuple(normalized_layers)

    if not layer_tuple:
        return ABLATION_STAGES[0]

    stage = next(
        (
            configuration
            for configuration in ABLATION_STAGES
            if configuration.layers == layer_tuple
        ),
        None,
    )

    if stage is None:
        raise AblationConfigurationError(
            "Ablation layers must form a sequential prefix "
            "from L1 through L5."
        )

    return stage


def build_ablation_progression(
    max_layers: int = 5,
) -> tuple[AblationConfiguration, ...]:
    """
    Build an ordered ablation progression.

    max_layers=0 returns only the no-defense baseline.
    max_layers=5 returns all six stages.
    """

    if isinstance(max_layers, bool) or not isinstance(max_layers, int):
        raise AblationConfigurationError(
            "max_layers must be an integer."
        )

    if max_layers < 0 or max_layers > len(DefenseLayer):
        raise AblationConfigurationError(
            f"max_layers must be between 0 and {len(DefenseLayer)}."
        )

    return ABLATION_STAGES[: max_layers + 1]


def is_layer_enabled(
    configuration: AblationConfiguration,
    layer: DefenseLayer | str,
) -> bool:
    """Return whether a defense layer is enabled in a configuration."""
    if not isinstance(configuration, AblationConfiguration):
        raise AblationConfigurationError(
            "configuration must be an AblationConfiguration."
        )

    if isinstance(layer, DefenseLayer):
        normalized_layer = layer
    elif isinstance(layer, str):
        try:
            normalized_layer = DefenseLayer(layer)
        except ValueError as exc:
            raise AblationConfigurationError(
                f"Unknown defense layer: {layer!r}."
            ) from exc
    else:
        raise AblationConfigurationError(
            "layer must be a DefenseLayer value or string."
        )

    return normalized_layer in configuration.layers


def compare_ablation_stages(
    baseline: AblationConfiguration,
    candidate: AblationConfiguration,
) -> dict[str, object]:
    """
    Describe the defense-layer difference between two configurations.

    This does not calculate security or utility metrics. Those belong
    to the later evaluation phase.
    """

    if not isinstance(baseline, AblationConfiguration):
        raise AblationConfigurationError(
            "baseline must be an AblationConfiguration."
        )

    if not isinstance(candidate, AblationConfiguration):
        raise AblationConfigurationError(
            "candidate must be an AblationConfiguration."
        )

    baseline_layers = set(baseline.layers)
    candidate_layers = set(candidate.layers)

    added = tuple(
        layer.value
        for layer in DefenseLayer
        if layer in candidate_layers - baseline_layers
    )

    removed = tuple(
        layer.value
        for layer in DefenseLayer
        if layer in baseline_layers - candidate_layers
    )

    return {
        "baseline": baseline.name,
        "candidate": candidate.name,
        "added_layers": added,
        "removed_layers": removed,
        "baseline_layer_count": baseline.layer_count,
        "candidate_layer_count": candidate.layer_count,
    }