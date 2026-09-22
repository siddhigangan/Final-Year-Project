from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from src.experiments.config_loader import (
    ExperimentConfigLoader,
    ExperimentConfigurationError,
    ExperimentFileConfig,
    ExperimentMatrixConfig,
)
from src.models import ExperimentCondition, ProgrammingLanguage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


class TestExperimentConfigLoaderInitialization:
    def test_loader_can_be_created_without_arguments(self) -> None:
        loader = ExperimentConfigLoader()

        assert isinstance(loader, ExperimentConfigLoader)


class TestExperimentFileLoading:
    def test_load_baseline_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "baseline.json")

        assert isinstance(config, ExperimentFileConfig)
        assert config.experiment_group == "baseline"
        assert config.version == "1.0"
        assert config.seed == 42
        assert len(config.conditions) == 1

        condition = config.conditions[0]

        assert condition.condition == ExperimentCondition.CLEAN_NO_DEFENSE
        assert condition.dataset.path == Path("data/clean")
        assert condition.poisoning.enabled is False
        assert condition.defense.enabled is False

    def test_load_poisoning_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "poisoning.json")

        assert isinstance(config, ExperimentFileConfig)
        assert config.experiment_group == "poisoning"
        assert len(config.conditions) == 1

        condition = config.conditions[0]

        assert condition.condition == ExperimentCondition.POISONED_NO_DEFENSE
        assert condition.dataset.path == Path("data/poisoned")
        assert condition.poisoning.enabled is True
        assert condition.poisoning.synthetic_only is True
        assert condition.poisoning.authorized_research_only is True
        assert condition.defense.enabled is False

    def test_load_defense_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")

        assert isinstance(config, ExperimentFileConfig)
        assert config.experiment_group == "defense"
        assert len(config.conditions) == 1

        condition = config.conditions[0]

        assert condition.condition == ExperimentCondition.POISONED_DEFENSE
        assert condition.dataset.path == Path("data/poisoned")
        assert condition.poisoning.enabled is True
        assert condition.defense.enabled is True

    def test_load_clean_defense_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "clean_defense.json")

        assert isinstance(config, ExperimentFileConfig)
        assert len(config.conditions) == 1

        condition = config.conditions[0]

        assert condition.condition == ExperimentCondition.CLEAN_DEFENSE
        assert condition.dataset.path == Path("data/clean")
        assert condition.poisoning.enabled is False
        assert condition.defense.enabled is True

    def test_all_four_condition_files_have_expected_conditions(self) -> None:
        loader = ExperimentConfigLoader()

        expected = {
            "baseline.json": ExperimentCondition.CLEAN_NO_DEFENSE,
            "poisoning.json": ExperimentCondition.POISONED_NO_DEFENSE,
            "defense.json": ExperimentCondition.POISONED_DEFENSE,
            "clean_defense.json": ExperimentCondition.CLEAN_DEFENSE,
        }

        for filename, expected_condition in expected.items():
            config = loader.load(EXPERIMENTS_DIR / filename)

            assert len(config.conditions) == 1
            assert config.conditions[0].condition == expected_condition


class TestDefenseConfiguration:
    def test_defense_configuration_contains_all_five_layers(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        defense = config.conditions[0].defense

        assert defense.enabled is True
        assert defense.fail_closed is True
        assert defense.build_safe_context is True
        assert defense.analyze_generated_code is True

        layer_names = [layer.name for layer in defense.layers]

        assert layer_names == [
            "source_trust_scoring",
            "anomaly_detection",
            "context_validation",
            "instruction_data_separation",
            "static_security_analysis",
        ]

    def test_defense_layer_keys_are_mapped_to_canonical_names(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        layers = config.conditions[0].defense.layers

        assert layers[0].name == "source_trust_scoring"
        assert layers[1].name == "anomaly_detection"
        assert layers[2].name == "context_validation"
        assert layers[3].name == "instruction_data_separation"
        assert layers[4].name == "static_security_analysis"

    def test_l1_has_expected_threshold(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        layer = config.conditions[0].defense.layers[0]

        assert layer.name == "source_trust_scoring"
        assert layer.enabled is True
        assert layer.threshold == pytest.approx(0.50)

    def test_l2_has_expected_threshold(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        layer = config.conditions[0].defense.layers[1]

        assert layer.name == "anomaly_detection"
        assert layer.enabled is True
        assert layer.threshold == pytest.approx(0.50)

    def test_clean_defense_enables_all_five_layers(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "clean_defense.json")
        defense = config.conditions[0].defense

        assert defense.enabled is True
        assert len(defense.layers) == 5
        assert all(layer.enabled for layer in defense.layers)


class TestPoisoningConfiguration:
    EXPECTED_CATEGORIES: ClassVar[set[str]] = {
        "misleading_code",
        "vulnerable_code",
        "false_api_guidance",
        "contradictory_documentation",
        "instruction_like_content",
        "false_repository_conventions",
        "context_manipulation",
    }

    def test_poisoning_contains_all_seven_categories(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "poisoning.json")
        categories = set(config.conditions[0].poisoning.categories)

        assert categories == self.EXPECTED_CATEGORIES

    def test_defense_configuration_contains_all_seven_categories(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        categories = set(config.conditions[0].poisoning.categories)

        assert categories == self.EXPECTED_CATEGORIES

    def test_poisoning_is_synthetic_and_authorized(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "poisoning.json")
        poisoning = config.conditions[0].poisoning

        assert poisoning.synthetic_only is True
        assert poisoning.authorized_research_only is True

    def test_clean_conditions_do_not_enable_poisoning(self) -> None:
        loader = ExperimentConfigLoader()

        for filename in ("baseline.json", "clean_defense.json"):
            config = loader.load(EXPERIMENTS_DIR / filename)
            poisoning = config.conditions[0].poisoning

            assert poisoning.enabled is False
            assert poisoning.categories == ()


class TestTaskLanguageConfiguration:
    EXPECTED_TASKS: ClassVar[set[str]] = {
        "code_generation",
        "code_completion",
        "bug_fixing",
        "test_generation",
        "api_usage",
        "repository_qa",
    }

    EXPECTED_LANGUAGES: ClassVar[set[ProgrammingLanguage]] = {
        ProgrammingLanguage.PYTHON,
        ProgrammingLanguage.JAVASCRIPT,
        ProgrammingLanguage.JAVA,
        ProgrammingLanguage.CPP,
    }

    def test_all_tasks_are_loaded(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "baseline.json")

        assert set(config.tasks) == self.EXPECTED_TASKS

    def test_all_languages_are_loaded(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "baseline.json")

        assert set(config.languages) == self.EXPECTED_LANGUAGES

    def test_tasks_are_shared_across_experiment_definitions(self) -> None:
        loader = ExperimentConfigLoader()

        filenames = (
            "baseline.json",
            "poisoning.json",
            "defense.json",
            "clean_defense.json",
        )

        for filename in filenames:
            config = loader.load(EXPERIMENTS_DIR / filename)

            assert set(config.tasks) == self.EXPECTED_TASKS

    def test_languages_are_shared_across_experiment_definitions(self) -> None:
        loader = ExperimentConfigLoader()

        filenames = (
            "baseline.json",
            "poisoning.json",
            "defense.json",
            "clean_defense.json",
        )

        for filename in filenames:
            config = loader.load(EXPERIMENTS_DIR / filename)

            assert set(config.languages) == self.EXPECTED_LANGUAGES


class TestRetrievalAndGenerationConfiguration:
    def test_retrieval_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "baseline.json")
        retrieval = config.retrieval

        assert retrieval.top_k == 5
        assert retrieval.retriever == "default"

    def test_generation_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "baseline.json")
        generation = config.generation

        assert generation.provider == "ollama"
        assert generation.temperature == pytest.approx(0.0)
        assert generation.max_tokens == 1024

    def test_evaluation_configuration_contains_required_metrics(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "poisoning.json")
        evaluation = config.evaluation

        assert evaluation.calculate_poison_retrieval_rate is True
        assert evaluation.calculate_attack_success_rate is True
        assert evaluation.calculate_vulnerability_introduction_rate is True
        assert evaluation.calculate_utility_score is True
        assert evaluation.track_generation_changes is True
        assert evaluation.run_security_analysis is True
        assert evaluation.track_causal_chain is True

    def test_defense_evaluation_tracks_defense_decisions(self) -> None:
        loader = ExperimentConfigLoader()

        config = loader.load(EXPERIMENTS_DIR / "defense.json")
        evaluation = config.evaluation

        assert evaluation.record_defense_decisions is True
        assert evaluation.record_blocked_context is True


class TestExperimentMatrix:
    def test_load_matrix(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert isinstance(matrix, ExperimentMatrixConfig)
        assert matrix.name == "CodeRAG-PoisonBench"
        assert matrix.version == "1.0"
        assert matrix.seed == 42

    def test_matrix_contains_all_four_conditions(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        conditions = [condition.condition for condition in matrix.conditions]

        assert conditions == [
            ExperimentCondition.CLEAN_NO_DEFENSE,
            ExperimentCondition.POISONED_NO_DEFENSE,
            ExperimentCondition.POISONED_DEFENSE,
            ExperimentCondition.CLEAN_DEFENSE,
        ]

    def test_matrix_condition_references_are_correct(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        references = {
            condition.condition: condition.configuration
            for condition in matrix.conditions
        }

        assert references[ExperimentCondition.CLEAN_NO_DEFENSE] == (
            "experiments/baseline.json"
        )
        assert references[ExperimentCondition.POISONED_NO_DEFENSE] == (
            "experiments/poisoning.json"
        )
        assert references[ExperimentCondition.POISONED_DEFENSE] == (
            "experiments/defense.json"
        )
        assert references[ExperimentCondition.CLEAN_DEFENSE] == (
            "experiments/clean_defense.json"
        )

    def test_matrix_has_expected_tasks(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert set(matrix.tasks) == {
            "code_generation",
            "code_completion",
            "bug_fixing",
            "test_generation",
            "api_usage",
            "repository_qa",
        }

    def test_matrix_has_expected_languages(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert set(matrix.languages) == {
            ProgrammingLanguage.PYTHON,
            ProgrammingLanguage.JAVASCRIPT,
            ProgrammingLanguage.JAVA,
            ProgrammingLanguage.CPP,
        }

    def test_matrix_replication_configuration(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert matrix.replication["repetitions"] == 3
        assert matrix.replication["independent_runs"] is True
        assert matrix.replication["seed_strategy"] == "fixed_base_seed"

    def test_matrix_contains_all_poisoning_categories(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert set(matrix.poisoning_categories) == {
            "misleading_code",
            "vulnerable_code",
            "false_api_guidance",
            "contradictory_documentation",
            "instruction_like_content",
            "false_repository_conventions",
            "context_manipulation",
        }

    def test_matrix_contains_expected_comparisons(self) -> None:
        loader = ExperimentConfigLoader()

        matrix = loader.load_matrix(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        comparison_names = {
            comparison["name"] for comparison in matrix.comparisons
        }

        assert comparison_names == {
            "poisoning_effect",
            "defense_effect_on_poisoned_data",
            "defense_utility_overhead",
        }


class TestLoadAll:
    def test_load_all_returns_matrix_and_condition_configs(self) -> None:
        loader = ExperimentConfigLoader()

        matrix, configurations = loader.load_all(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert isinstance(matrix, ExperimentMatrixConfig)
        assert isinstance(configurations, dict)

        assert set(configurations) == {
            "A_clean_no_defense",
            "B_poisoned_no_defense",
            "C_poisoned_defense",
            "D_clean_defense",
        }

    def test_load_all_returns_correct_configuration_types(self) -> None:
        loader = ExperimentConfigLoader()

        _, configurations = loader.load_all(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert all(
            isinstance(config, ExperimentFileConfig)
            for config in configurations.values()
        )

    def test_load_all_condition_mapping_is_correct(self) -> None:
        loader = ExperimentConfigLoader()

        _, configurations = loader.load_all(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert (
            configurations["A_clean_no_defense"].conditions[0].condition
            == ExperimentCondition.CLEAN_NO_DEFENSE
        )
        assert (
            configurations["B_poisoned_no_defense"].conditions[0].condition
            == ExperimentCondition.POISONED_NO_DEFENSE
        )
        assert (
            configurations["C_poisoned_defense"].conditions[0].condition
            == ExperimentCondition.POISONED_DEFENSE
        )
        assert (
            configurations["D_clean_defense"].conditions[0].condition
            == ExperimentCondition.CLEAN_DEFENSE
        )

    def test_load_all_defense_mapping_is_correct(self) -> None:
        loader = ExperimentConfigLoader()

        _, configurations = loader.load_all(
            EXPERIMENTS_DIR / "experiment_matrix.json"
        )

        assert (
            configurations["C_poisoned_defense"]
            .conditions[0]
            .defense
            .enabled
            is True
        )
        assert (
            configurations["D_clean_defense"]
            .conditions[0]
            .defense
            .enabled
            is True
        )
        assert (
            configurations["B_poisoned_no_defense"]
            .conditions[0]
            .defense
            .enabled
            is False
        )


class TestInvalidConfigurations:
    @staticmethod
    def _write_json(
        tmp_path: Path,
        filename: str,
        payload: dict,
    ) -> Path:
        path = tmp_path / filename
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _base_payload(
        condition: str = "A_clean_no_defense",
    ) -> dict:
        return {
            "experiment_group": "test",
            "description": "Test configuration",
            "version": "1.0",
            "seed": 42,
            "conditions": [
                {
                    "condition": condition,
                    "dataset": {
                        "type": (
                            "poisoned"
                            if condition
                            in {
                                "B_poisoned_no_defense",
                                "C_poisoned_defense",
                            }
                            else "clean"
                        ),
                        "path": (
                            "data/poisoned"
                            if condition
                            in {
                                "B_poisoned_no_defense",
                                "C_poisoned_defense",
                            }
                            else "data/clean"
                        ),
                    },
                    "poisoning": {
                        "enabled": condition
                        in {
                            "B_poisoned_no_defense",
                            "C_poisoned_defense",
                        },
                        "categories": (
                            ["misleading_code"]
                            if condition
                            in {
                                "B_poisoned_no_defense",
                                "C_poisoned_defense",
                            }
                            else []
                        ),
                        "synthetic_only": True,
                        "authorized_research_only": True,
                    },
                    "defense": {
                        "enabled": condition
                        in {
                            "C_poisoned_defense",
                            "D_clean_defense",
                        },
                        "layers": [],
                    },
                }
            ],
            "tasks": ["code_generation"],
            "languages": ["python"],
            "retrieval": {
                "top_k": 5,
                "retriever": "default",
            },
            "generation": {
                "provider": "ollama",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            "evaluation": {},
        }

    def test_missing_configuration_file_raises(
        self,
        tmp_path: Path,
    ) -> None:
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "invalid.json"
        path.write_text("{invalid json", encoding="utf-8")

        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_missing_experiment_group_raises(self, tmp_path: Path) -> None:
        payload = self._base_payload()
        del payload["experiment_group"]

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_condition_raises(self, tmp_path: Path) -> None:
        payload = self._base_payload()
        payload["conditions"][0]["condition"] = "invalid_condition"

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_dataset_type_raises(self, tmp_path: Path) -> None:
        payload = self._base_payload()
        payload["conditions"][0]["dataset"]["type"] = "unknown"

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_poisoning_category_raises(
        self,
        tmp_path: Path,
    ) -> None:
        payload = self._base_payload(
            "B_poisoned_no_defense",
        )
        payload["conditions"][0]["poisoning"]["categories"] = [
            "not_a_real_category"
        ]

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_task_raises(self, tmp_path: Path) -> None:
        payload = self._base_payload()
        payload["tasks"] = ["not_a_real_task"]

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_language_raises(self, tmp_path: Path) -> None:
        payload = self._base_payload()
        payload["languages"] = ["not_a_real_language"]

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_invalid_generation_provider_raises(
        self,
        tmp_path: Path,
    ) -> None:
        payload = self._base_payload()
        payload["generation"]["provider"] = "invalid_provider"

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_defense_with_missing_layer_raises(
        self,
        tmp_path: Path,
    ) -> None:
        payload = self._base_payload(
            "C_poisoned_defense",
        )

        payload["conditions"][0]["defense"] = {
            "enabled": True,
            "layers": {
                "L1": {
                    "name": "source_trust_scoring",
                    "enabled": True,
                    "threshold": 0.50,
                }
            },
        }

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)

    def test_defense_layer_name_mismatch_raises(
        self,
        tmp_path: Path,
    ) -> None:
        payload = self._base_payload(
            "C_poisoned_defense",
        )

        payload["conditions"][0]["defense"] = {
            "enabled": True,
            "layers": {
                "L1": {
                    "name": "anomaly_detection",
                    "enabled": True,
                    "threshold": 0.50,
                },
                "L2": {
                    "name": "anomaly_detection",
                    "enabled": True,
                    "threshold": 0.50,
                },
                "L3": {
                    "name": "context_validation",
                    "enabled": True,
                },
                "L4": {
                    "name": "instruction_data_separation",
                    "enabled": True,
                },
                "L5": {
                    "name": "static_security_analysis",
                    "enabled": True,
                },
            },
        }

        path = self._write_json(tmp_path, "invalid.json", payload)
        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load(path)


class TestMatrixValidation:
    @staticmethod
    def _write_json(
        tmp_path: Path,
        filename: str,
        payload: dict,
    ) -> Path:
        path = tmp_path / filename
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _matrix_payload(
        configuration: str,
        condition: str = "A_clean_no_defense",
    ) -> dict:
        return {
            "experiment_matrix": {
                "name": "TestMatrix",
                "version": "1.0",
                "description": "Test matrix",
                "research_scope": (
                    "authorized_synthetic_code_rag_security"
                ),
                "synthetic_only": True,
                "authorized_research_only": True,
                "seed": 42,
            },
            "conditions": [
                {
                    "id": "A",
                    "name": "clean_no_defense",
                    "condition": condition,
                    "dataset": "clean",
                    "defense_enabled": False,
                    "poisoning_enabled": False,
                    "configuration": configuration,
                }
            ],
            "tasks": ["code_generation"],
            "languages": ["python"],
            "retrieval": {
                "top_k": 5,
                "retriever": "default",
            },
            "generation": {
                "provider": "ollama",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            "replication": {
                "repetitions": 1,
                "independent_runs": True,
                "seed_strategy": "fixed_base_seed",
            },
            "evaluation": {},
            "poisoning_categories": [],
            "comparisons": [],
            "output": {
                "result_store": "results/experiments.jsonl",
                "benchmark_store": "results/benchmark_runs.jsonl",
                "report_directory": "results",
                "preserve_raw_results": True,
            },
            "metadata": {},
        }

    def test_matrix_with_unknown_configuration_reference_raises(
        self,
        tmp_path: Path,
    ) -> None:
        matrix = self._matrix_payload("missing.json")

        matrix_path = self._write_json(
            tmp_path,
            "experiment_matrix.json",
            matrix,
        )

        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load_all(matrix_path)

    def test_matrix_condition_must_match_referenced_configuration(
        self,
        tmp_path: Path,
    ) -> None:
        config_payload = {
            "experiment_group": "test",
            "description": "Test configuration",
            "version": "1.0",
            "seed": 42,
            "conditions": [
                {
                    "condition": "B_poisoned_no_defense",
                    "dataset": {
                        "type": "poisoned",
                        "base_path": "data/clean",
                        "output_path": "data/poisoned",
                    },
                    "poisoning": {
                        "enabled": True,
                        "categories": ["misleading_code"],
                        "synthetic_only": True,
                        "authorized_research_only": True,
                    },
                    "defense": {
                        "enabled": False,
                        "layers": [],
                    },
                }
            ],
            "tasks": ["code_generation"],
            "languages": ["python"],
            "retrieval": {
                "top_k": 5,
                "retriever": "default",
            },
            "generation": {
                "provider": "ollama",
                "temperature": 0.0,
                "max_tokens": 1024,
            },
            "evaluation": {},
        }

        self._write_json(
            tmp_path,
            "config.json",
            config_payload,
        )

        matrix = self._matrix_payload(
            "config.json",
            condition="A_clean_no_defense",
        )

        matrix_path = self._write_json(
            tmp_path,
            "matrix.json",
            matrix,
        )

        loader = ExperimentConfigLoader()

        with pytest.raises(ExperimentConfigurationError):
            loader.load_all(matrix_path)