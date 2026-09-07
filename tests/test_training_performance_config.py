import pytest
import torch
from torch_geometric.data import Data

from src.benchmark_chess_gat_training import (
    BenchmarkConfig,
    classify_bottleneck,
    estimate_epoch_times,
    make_loader,
    run_single_benchmark,
    select_recommended_config,
)
from src.training.chess_gat_trainer import (
    ChessGATTrainingConfig,
    make_loaders,
    set_loader_epoch,
)


def _graph(target=0):
    return Data(
        x=torch.zeros((64, 15), dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=torch.float,
        ),
        global_features=torch.zeros((1, 4), dtype=torch.float),
        y=torch.tensor(target, dtype=torch.long),
    )


def _graphs(count=4):
    return [
        _graph(index % 2)
        for index in range(count)
    ]


def test_make_loaders_rejects_invalid_worker_options():
    config = ChessGATTrainingConfig(
        num_workers=0,
        persistent_workers=True,
    )

    with pytest.raises(ValueError, match="persistent_workers"):
        make_loaders(
            _graphs(),
            _graphs(),
            _graphs(),
            config,
        )

    config = ChessGATTrainingConfig(
        num_workers=0,
        prefetch_factor=2,
    )

    with pytest.raises(ValueError, match="prefetch_factor"):
        make_loaders(
            _graphs(),
            _graphs(),
            _graphs(),
            config,
        )


def test_make_loaders_accepts_runtime_options_with_workers_zero():
    config = ChessGATTrainingConfig(
        num_workers=0,
        pin_memory=False,
        non_blocking=True,
        amp=True,
    )

    train_loader, val_loader, test_loader = make_loaders(
        _graphs(),
        _graphs(),
        _graphs(),
        config,
    )

    assert train_loader.num_workers == 0
    assert val_loader.num_workers == 0
    assert test_loader.num_workers == 0


def test_set_loader_epoch_updates_sampler_when_supported():
    class _Sampler:
        def __init__(self):
            self.epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

    class _Loader:
        def __init__(self):
            self.sampler = _Sampler()

    loader = _Loader()

    set_loader_epoch(loader, 7)

    assert loader.sampler.epoch == 7


def test_benchmark_loader_rejects_invalid_prefetch_factor():
    config = BenchmarkConfig(
        batch_size=2,
        num_workers=0,
        amp=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        non_blocking=False,
        sampler="standard",
    )

    with pytest.raises(ValueError, match="prefetch_factor"):
        make_loader(
            _graphs(),
            config,
        )


def test_estimate_epoch_times_uses_full_train_count():
    estimates = estimate_epoch_times(
        train_graphs=1000,
        graphs_per_second=100,
    )

    assert estimates["estimated_full_train_epoch_seconds"] == 10
    assert estimates["estimated_epoch_minutes"] == pytest.approx(10 / 60)
    assert estimates["estimated_30_epochs_hours"] == pytest.approx(300 / 3600)
    assert estimates["estimated_120_epochs_hours"] == pytest.approx(1200 / 3600)


def test_cpu_amp_configuration_is_marked_not_applicable():
    config = BenchmarkConfig(
        batch_size=2,
        num_workers=0,
        amp=True,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        non_blocking=False,
        sampler="standard",
    )

    result = run_single_benchmark(
        _graphs(),
        4,
        num_classes=2,
        device=torch.device("cpu"),
        config=config,
        warmup_batches=1,
        benchmark_batches=1,
        seed=42,
    )

    assert result["status"] == "AMP_NOT_APPLICABLE"


def test_bottleneck_classifier_and_recommendation_are_stable():
    results = [
        {
            "status": "OK",
            "batch_size": 32,
            "num_workers": 0,
            "amp": False,
            "sampler": "shard_aware",
            "graphs_per_second": 100.0,
            "mean_data_seconds_per_batch": 0.01,
            "mean_compute_seconds_per_batch": 0.10,
            "peak_gpu_memory_reserved_mb": 100.0,
        },
        {
            "status": "OK",
            "batch_size": 64,
            "num_workers": 2,
            "amp": False,
            "sampler": "shard_aware",
            "graphs_per_second": 102.0,
            "mean_data_seconds_per_batch": 0.02,
            "mean_compute_seconds_per_batch": 0.08,
            "peak_gpu_memory_reserved_mb": 90.0,
        },
        {
            "status": "OOM",
            "batch_size": 256,
            "num_workers": 4,
            "amp": True,
            "sampler": "shard_aware",
        },
    ]

    diagnosis = classify_bottleneck(results)
    recommendation = select_recommended_config(results)

    assert diagnosis["verdict"] in {
        "GPU_COMPUTE_BOUND",
        "DATALOADER_BOUND",
        "SHARD_IO_BOUND",
        "MIXED",
        "INCONCLUSIVE",
    }
    assert recommendation["batch_size"] == 64
