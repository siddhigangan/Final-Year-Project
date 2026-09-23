from src.benchmark.dataset import (
    BenchmarkDataset,
    BenchmarkDatasetInputError,
    BenchmarkSample,
)


def make_sample(sample_id="sample-1"):
    return BenchmarkSample(
        sample_id=sample_id,
        query="How should this code be secured?",
        clean_code="safe_code()",
        poisoned_code="safe_code_with_marker()",
        language="python",
        poisoning_category="misleading_code",
    )


def test_sample_is_modified():
    sample = make_sample()

    assert sample.is_modified is True


def test_dataset_add_and_get():
    dataset = BenchmarkDataset()
    sample = make_sample()

    dataset.add(sample)

    assert dataset.size == 1
    assert dataset.get("sample-1") == sample


def test_dataset_rejects_duplicate_ids():
    dataset = BenchmarkDataset()
    dataset.add(make_sample())

    try:
        dataset.add(make_sample())
    except BenchmarkDatasetInputError:
        return

    raise AssertionError("Expected duplicate ID rejection")


def test_categories():
    dataset = BenchmarkDataset(
        [
            make_sample("1"),
            BenchmarkSample(
                sample_id="2",
                query="q",
                clean_code="clean",
                poisoned_code="poisoned",
                language="python",
                poisoning_category="vulnerable_code",
            ),
        ]
    )

    assert dataset.categories() == [
        "misleading_code",
        "vulnerable_code",
    ]