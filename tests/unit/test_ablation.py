from __future__ import annotations

import pytest

from src.experiments.ablation import (
    ABLATION_STAGES,
    AblationConfiguration,
    AblationConfigurationError,
    DefenseLayer,
    build_ablation_configuration,
    build_ablation_progression,
    compare_ablation_stages,
    get_ablation_stage,
    get_ablation_stages,
    is_layer_enabled,
)


class TestAblationConfiguration:
    def test_no_defense_configuration(self) -> None:
        configuration = AblationConfiguration(
            name="no_defense",
            layers=(),
        )

        assert configuration.name == "no_defense"
        assert configuration.enabled_layers == ()
        assert configuration.layer_count == 0
        assert configuration.is_baseline is True

    def test_single_layer_configuration(self) -> None:
        configuration = AblationConfiguration(
            name="L1",
            layers=(DefenseLayer.L1,),
        )

        assert configuration.enabled_layers == ("L1",)
        assert configuration.layer_count == 1
        assert configuration.is_baseline is False

    def test_full_defense_configuration(self) -> None:
        configuration = AblationConfiguration(
            name="L1_L2_L3_L4_L5",
            layers=(
                DefenseLayer.L1,
                DefenseLayer.L2,
                DefenseLayer.L3,
                DefenseLayer.L4,
                DefenseLayer.L5,
            ),
        )

        assert configuration.layer_count == 5
        assert configuration.enabled_layers == (
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        )

    def test_configuration_serialization(self) -> None:
        configuration = AblationConfiguration(
            name="L1_L2",
            layers=(
                DefenseLayer.L1,
                DefenseLayer.L2,
            ),
        )

        assert configuration.as_dict() == {
            "name": "L1_L2",
            "layers": ["L1", "L2"],
            "layer_count": 2,
            "is_baseline": False,
        }


class TestAblationValidation:
    def test_empty_name_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="",
                layers=(),
            )

    def test_non_string_name_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name=123,  # type: ignore[arg-type]
                layers=(),
            )

    def test_layers_must_be_tuple(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="L1",
                layers=[DefenseLayer.L1],  # type: ignore[arg-type]
            )

    def test_invalid_layer_type_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="invalid",
                layers=("L1",),  # type: ignore[arg-type]
            )

    def test_duplicate_layers_are_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="invalid",
                layers=(
                    DefenseLayer.L1,
                    DefenseLayer.L1,
                ),
            )

    def test_non_sequential_layers_are_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="invalid",
                layers=(
                    DefenseLayer.L1,
                    DefenseLayer.L3,
                ),
            )

    def test_l2_without_l1_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="invalid",
                layers=(DefenseLayer.L2,),
            )

    def test_l5_without_previous_layers_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            AblationConfiguration(
                name="invalid",
                layers=(DefenseLayer.L5,),
            )


class TestAblationStages:
    def test_all_six_stages_exist(self) -> None:
        stages = get_ablation_stages()

        assert len(stages) == 6

    def test_stages_are_ordered(self) -> None:
        stages = get_ablation_stages()

        assert [stage.name for stage in stages] == [
            "no_defense",
            "L1",
            "L1_L2",
            "L1_L2_L3",
            "L1_L2_L3_L4",
            "L1_L2_L3_L4_L5",
        ]

    def test_stage_layer_counts_increase_sequentially(self) -> None:
        stages = get_ablation_stages()

        assert [stage.layer_count for stage in stages] == [
            0,
            1,
            2,
            3,
            4,
            5,
        ]

    def test_global_stage_definition_is_immutable(self) -> None:
        assert ABLATION_STAGES == get_ablation_stages()


class TestAblationLookup:
    def test_get_no_defense_stage(self) -> None:
        stage = get_ablation_stage("no_defense")

        assert stage.layer_count == 0
        assert stage.is_baseline is True

    def test_get_full_defense_stage(self) -> None:
        stage = get_ablation_stage("L1_L2_L3_L4_L5")

        assert stage.enabled_layers == (
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        )

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            get_ablation_stage("unknown")

    def test_empty_stage_name_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            get_ablation_stage("")


class TestAblationBuilder:
    def test_build_empty_configuration(self) -> None:
        configuration = build_ablation_configuration([])

        assert configuration.name == "no_defense"

    def test_build_from_enum(self) -> None:
        configuration = build_ablation_configuration(
            [
                DefenseLayer.L1,
                DefenseLayer.L2,
            ]
        )

        assert configuration.name == "L1_L2"

    def test_build_from_strings(self) -> None:
        configuration = build_ablation_configuration(
            ["L1", "L2", "L3"]
        )

        assert configuration.name == "L1_L2_L3"

    def test_invalid_string_layer_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_configuration(["L6"])

    def test_invalid_layer_type_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_configuration([123])  # type: ignore[list-item]

    def test_non_sequential_layers_raise(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_configuration(["L1", "L3"])


class TestAblationProgression:
    def test_default_progression_contains_all_stages(self) -> None:
        progression = build_ablation_progression()

        assert len(progression) == 6

    def test_zero_layers_returns_baseline(self) -> None:
        progression = build_ablation_progression(0)

        assert len(progression) == 1
        assert progression[0].name == "no_defense"

    def test_three_layers_returns_four_stages(self) -> None:
        progression = build_ablation_progression(3)

        assert [stage.layer_count for stage in progression] == [
            0,
            1,
            2,
            3,
        ]

    def test_five_layers_returns_all_stages(self) -> None:
        progression = build_ablation_progression(5)

        assert progression[-1].name == "L1_L2_L3_L4_L5"

    def test_negative_max_layers_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_progression(-1)

    def test_too_many_layers_raises(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_progression(6)

    def test_boolean_max_layers_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            build_ablation_progression(True)  # type: ignore[arg-type]


class TestLayerEnabled:
    def test_enabled_layer_returns_true(self) -> None:
        configuration = get_ablation_stage("L1_L2_L3")

        assert is_layer_enabled(
            configuration,
            DefenseLayer.L2,
        ) is True

    def test_disabled_layer_returns_false(self) -> None:
        configuration = get_ablation_stage("L1_L2")

        assert is_layer_enabled(
            configuration,
            DefenseLayer.L3,
        ) is False

    def test_string_layer_is_supported(self) -> None:
        configuration = get_ablation_stage("L1_L2")

        assert is_layer_enabled(configuration, "L2") is True

    def test_invalid_configuration_is_rejected(self) -> None:
        with pytest.raises(AblationConfigurationError):
            is_layer_enabled(
                "L1",  # type: ignore[arg-type]
                DefenseLayer.L1,
            )


class TestAblationComparison:
    def test_comparison_identifies_added_layer(self) -> None:
        baseline = get_ablation_stage("L1")
        candidate = get_ablation_stage("L1_L2")

        result = compare_ablation_stages(
            baseline,
            candidate,
        )

        assert result["added_layers"] == ("L2",)
        assert result["removed_layers"] == ()

    def test_comparison_identifies_multiple_added_layers(self) -> None:
        baseline = get_ablation_stage("no_defense")
        candidate = get_ablation_stage("L1_L2_L3")

        result = compare_ablation_stages(
            baseline,
            candidate,
        )

        assert result["added_layers"] == (
            "L1",
            "L2",
            "L3",
        )

    def test_comparison_identifies_removed_layer(self) -> None:
        baseline = get_ablation_stage("L1_L2")
        candidate = get_ablation_stage("L1")

        result = compare_ablation_stages(
            baseline,
            candidate,
        )

        assert result["added_layers"] == ()
        assert result["removed_layers"] == ("L2",)

    def test_comparison_contains_layer_counts(self) -> None:
        baseline = get_ablation_stage("L1")
        candidate = get_ablation_stage("L1_L2_L3")

        result = compare_ablation_stages(
            baseline,
            candidate,
        )

        assert result["baseline_layer_count"] == 1
        assert result["candidate_layer_count"] == 3

    def test_invalid_baseline_is_rejected(self) -> None:
        candidate = get_ablation_stage("L1")

        with pytest.raises(AblationConfigurationError):
            compare_ablation_stages(
                "invalid",  # type: ignore[arg-type]
                candidate,
            )

    def test_invalid_candidate_is_rejected(self) -> None:
        baseline = get_ablation_stage("L1")

        with pytest.raises(AblationConfigurationError):
            compare_ablation_stages(
                baseline,
                "invalid",  # type: ignore[arg-type]
            )