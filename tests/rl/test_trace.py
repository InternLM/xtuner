import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from xtuner.tools.producer_trace_analysis import load_trace_jsonl
from xtuner.tools.producer_trace_hotspots import build_hotspot_payload_from_events
from xtuner.tools.producer_trace_viewer import build_viewer_payload_from_events
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.trace import (
    InMemoryTraceStore,
    TraceConfig,
    TraceEvent,
    TraceRecorder,
    close_trace,
    configure_trace,
    get_trace_env_vars,
    merge_trace_runtime_env,
    reset_trace_for_test,
    trace_event,
    trace_function,
    trace_span,
    use_trace_recorder,
)


def make_state(uid: int = 1, task_name: str = "gsm8k", status: Status = Status.INIT) -> RolloutState:
    return RolloutState(
        message=[{"role": "user", "content": "What is 1 + 1?"}],
        uid=uid,
        task_name=task_name,
        session_uid=uid + 1000,
        status=status,
    )


def make_event(
    trace_id: str,
    stage: str,
    timestamp_s: float,
    *,
    task_name: str = "gsm8k",
    uid: int = 1,
    status: str = "init",
    elapsed_s: float | None = None,
    train_step: int | None = None,
    model_step: int | None = None,
    producer_future_step: int | None = None,
    produce_batch_id: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        trace_id=trace_id,
        stage=stage,
        timestamp_s=timestamp_s,
        status=status,
        task_name=task_name,
        uid=uid,
        session_uid=uid + 1000,
        train_step=train_step,
        model_step=model_step,
        producer_future_step=producer_future_step,
        produce_batch_id=produce_batch_id,
        worker_rank=None,
        elapsed_s=elapsed_s,
        error_msg=None,
    )


def make_batch_event(
    trace_id: str,
    stage: str,
    timestamp_s: float,
    *,
    uid: int,
    batch: str,
    train_step: int,
    model_step: int = 0,
    producer_future_step: int = 0,
    elapsed_s: float | None = None,
    status: str = "init",
) -> TraceEvent:
    return make_event(
        trace_id,
        stage,
        timestamp_s,
        uid=uid,
        status=status,
        elapsed_s=elapsed_s,
        train_step=train_step,
        model_step=model_step,
        producer_future_step=producer_future_step,
        produce_batch_id=batch,
    )


class TraceCoreBehaviorTest(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        reset_trace_for_test()

    async def test_trace_api_records_event_span_function_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = InMemoryTraceStore(
                TraceConfig(enabled=True, output_dir=tmp_dir, max_events=20, max_events_per_trace=20)
            )
            recorder = TraceRecorder(store)
            state = make_state(uid=123)

            with use_trace_recorder(recorder):
                await trace_event(state, "custom.prepare")

                async with trace_span(state, "custom.work"):
                    state.status = Status.COMPLETED

                @trace_function("custom.fn")
                async def traced_fn(rollout_state: RolloutState) -> RolloutState:
                    await trace_event(rollout_state, "custom.fn.inner")
                    return rollout_state

                await traced_fn(state)

            store.flush_jsonl()
            store.close()
            timeline = store.get_timeline("gsm8k:123")
            disk_events = load_trace_jsonl(tmp_dir)

        expected_stages = [
            "custom.prepare",
            "custom.work.start",
            "custom.work.end",
            "custom.fn.start",
            "custom.fn.inner",
            "custom.fn.end",
        ]
        self.assertEqual([event.stage for event in timeline], expected_stages)
        self.assertEqual([event.stage for event in disk_events], expected_stages)
        self.assertEqual(timeline[-1].status, "completed")

    async def test_trace_function_resolves_target_and_dynamic_kwargs(self):
        store = InMemoryTraceStore(TraceConfig(enabled=True, max_events=10, max_events_per_trace=10))
        batch_id = "train_step=1/model_step=2/producer_future_step=3"

        class Worker:
            def __init__(self):
                self.worker_rank = 7

            @trace_function(
                "custom.worker",
                target="state",
                trace_kwargs_getter=lambda self, *args, **kwargs: {
                    "produce_batch_id": batch_id,
                    "worker_rank": self.worker_rank,
                },
            )
            async def run(self, state: RolloutState) -> RolloutState:
                return state.model_copy(update={"status": Status.COMPLETED}, deep=True)

        state = make_state(uid=456)
        with use_trace_recorder(TraceRecorder(store)):
            await Worker().run(state)
        timeline = store.get_timeline("gsm8k:456")

        self.assertEqual([event.stage for event in timeline], ["custom.worker.start", "custom.worker.end"])
        self.assertEqual([event.produce_batch_id for event in timeline], [batch_id, batch_id])
        self.assertEqual([event.worker_rank for event in timeline], [7, 7])
        self.assertEqual(timeline[0].status, "init")
        self.assertEqual(timeline[-1].status, "completed")

    async def test_trace_function_records_error_event_and_reraises(self):
        store = InMemoryTraceStore(TraceConfig(enabled=True, max_events=10, max_events_per_trace=10))
        state = make_state(uid=789)

        @trace_function("custom.failure", target="state")
        async def failing_fn(state: RolloutState) -> None:
            raise ValueError("rollout failed")

        with use_trace_recorder(TraceRecorder(store)):
            with self.assertRaisesRegex(ValueError, "rollout failed"):
                await failing_fn(state)

        timeline = store.get_timeline("gsm8k:789")
        self.assertEqual([event.stage for event in timeline], ["custom.failure.start", "custom.failure.error"])
        self.assertIsNotNone(timeline[-1].elapsed_s)
        self.assertTrue(timeline[-1].error_msg.startswith("ValueError: rollout failed"))

    def test_trace_runtime_env_is_propagated_to_ray_actor_options(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            configure_trace(TraceConfig(enabled=True, output_dir=tmp_dir, max_events=20, max_events_per_trace=30))
            actor_options = {"num_cpus": 1, "runtime_env": {"env_vars": {"EXISTING": "1"}}}

            env_vars = get_trace_env_vars()
            merged = merge_trace_runtime_env(actor_options)
            close_trace()

        self.assertIs(merged, actor_options)
        self.assertEqual(env_vars["XTUNER_TRACE_ENABLED"], "1")
        self.assertEqual(env_vars["XTUNER_TRACE_OUTPUT_DIR"], str(Path(tmp_dir).absolute()))
        self.assertEqual(env_vars["XTUNER_TRACE_MAX_EVENTS"], "20")
        self.assertEqual(env_vars["XTUNER_TRACE_MAX_EVENTS_PER_TRACE"], "30")
        self.assertEqual(actor_options["runtime_env"]["env_vars"]["EXISTING"], "1")
        self.assertEqual(actor_options["runtime_env"]["env_vars"]["XTUNER_TRACE_ENABLED"], "1")
        self.assertEqual(actor_options["runtime_env"]["env_vars"]["XTUNER_TRACE_OUTPUT_DIR"], str(Path(tmp_dir).absolute()))
        self.assertEqual(get_trace_env_vars(), {})

    def test_online_viewer_payload_reports_latest_batch_and_current_stalls(self):
        old_batch = "train_step=1/model_step=0/producer_future_step=0"
        latest_batch = "train_step=2/model_step=0/producer_future_step=0"
        events = [
            make_batch_event(
                "gsm8k:1",
                "xtuner.rollout_controller.generate.start",
                5.0,
                uid=1,
                batch=old_batch,
                train_step=1,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_controller.generate.start",
                10.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:3",
                "xtuner.judger.judge.start",
                12.0,
                uid=3,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:4",
                "xtuner.producer.put_generated_group.start",
                13.0,
                uid=4,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:4",
                "xtuner.producer.put_generated_group.end",
                15.0,
                uid=4,
                batch=latest_batch,
                train_step=2,
                elapsed_s=2.0,
                status="completed",
            ),
        ]

        with patch("xtuner.tools.producer_trace_analysis.time.time", return_value=30.0):
            payload = build_viewer_payload_from_events(events, trace_source="/tmp/trace")

        summary = payload["task_summary"]
        self.assertEqual(payload["raw_event_count"], 5)
        self.assertEqual(payload["event_count"], 4)
        self.assertEqual(summary["total_tasks"], 3)
        self.assertEqual(summary["running_tasks"], 2)
        self.assertEqual(summary["completed_tasks"], 1)
        self.assertEqual(summary["current_stage_counts"]["rollout.generate"], 1)
        self.assertEqual(summary["current_stage_counts"]["judger"], 1)
        self.assertEqual({row["trace_id"] for row in payload["rows"]}, {"gsm8k:2", "gsm8k:3", "gsm8k:4"})

        open_summary_by_span = {item["span"]: item for item in payload["open_span_summaries"]}
        self.assertEqual(open_summary_by_span["xtuner.rollout_controller.generate"]["oldest_trace_id"], "gsm8k:2")
        self.assertEqual(open_summary_by_span["xtuner.rollout_controller.generate"]["oldest_age_s"], 20.0)
        self.assertEqual(open_summary_by_span["xtuner.judger.judge"]["oldest_age_s"], 18.0)

    def test_hotspot_payload_builds_nested_spans_and_stage_stats(self):
        old_batch = "train_step=1/model_step=0/producer_future_step=0"
        latest_batch = "train_step=2/model_step=0/producer_future_step=0"
        events = [
            make_batch_event(
                "gsm8k:1",
                "xtuner.producer.generate_group.start",
                0.0,
                uid=1,
                batch=old_batch,
                train_step=1,
            ),
            make_batch_event(
                "gsm8k:1",
                "xtuner.producer.generate_group.end",
                1.0,
                uid=1,
                batch=old_batch,
                train_step=1,
                elapsed_s=1.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.producer.generate_group.start",
                100.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.agent_loop.generate_group.start",
                101.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.agent_loop.generate_sample.start",
                102.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_controller.generate.start",
                103.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_worker.generate.start",
                104.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_engine.generate.start",
                105.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_engine.generate.end",
                108.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=3.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_worker.generate.end",
                109.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=5.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.rollout_controller.generate.end",
                110.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=7.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.judger.judge.start",
                111.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.judger.judge.end",
                115.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=4.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.agent_loop.generate_sample.end",
                116.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=14.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.agent_loop.generate_group.end",
                117.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=16.0,
            ),
            make_batch_event(
                "gsm8k:2",
                "xtuner.producer.generate_group.end",
                118.0,
                uid=2,
                batch=latest_batch,
                train_step=2,
                elapsed_s=18.0,
            ),
            make_batch_event(
                "gsm8k:3",
                "xtuner.rollout_controller.generate.start",
                1000.0,
                uid=3,
                batch=latest_batch,
                train_step=2,
            ),
            make_batch_event(
                "gsm8k:3",
                "xtuner.rollout_controller.generate.end",
                1010.0,
                uid=3,
                batch=latest_batch,
                train_step=2,
                elapsed_s=10.0,
            ),
        ]

        payload = build_hotspot_payload_from_events(events, trace_source="/tmp/trace")
        all_payload = build_hotspot_payload_from_events(events, trace_source="/tmp/trace", scope="all")

        self.assertEqual(payload["task_count"], 2)
        self.assertEqual(all_payload["task_count"], 3)
        self.assertEqual(payload["scale_mode"], "task_relative")

        row_by_trace_id = {row["trace_id"]: row for row in payload["rows"]}
        nested_spans = row_by_trace_id["gsm8k:2"]["spans"]
        self.assertEqual(
            [span["display_stage"] for span in nested_spans],
            [
                "producer.generate",
                "agent_loop.generate_group",
                "agent_loop.generate_sample",
                "rollout.generate",
                "rollout_worker.generate",
                "engine.generate",
                "judger",
            ],
        )
        self.assertEqual([span["depth"] for span in nested_spans], [0, 1, 2, 3, 4, 5, 3])
        self.assertEqual(nested_spans[0]["left_pct"], 0.0)
        self.assertEqual(row_by_trace_id["gsm8k:3"]["spans"][0]["left_pct"], 0.0)

        stats_by_stage = {stat["stage"]: stat for stat in payload["stage_stats"]}
        self.assertEqual(stats_by_stage["engine.generate"]["avg_s"], 3.0)
        self.assertEqual(stats_by_stage["engine.generate"]["p95_s"], 3.0)
        self.assertEqual(stats_by_stage["engine.generate"]["max_s"], 3.0)
        self.assertEqual(stats_by_stage["rollout.generate"]["count"], 2)
        self.assertEqual(stats_by_stage["rollout.generate"]["avg_s"], 8.5)
        self.assertEqual(stats_by_stage["rollout.generate"]["max_s"], 10.0)


class TraceTrainerIntegrationTest(unittest.TestCase):
    def tearDown(self):
        reset_trace_for_test()

    def test_trainer_starts_live_viewer_on_rank0(self):
        with patch.dict("sys.modules", {"causal_conv1d_cuda": MagicMock()}):
            from xtuner.v1.train import rl_trainer

        BaseRLTrainer = rl_trainer.BaseRLTrainer

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = object.__new__(BaseRLTrainer)
            trainer._meta = SimpleNamespace(latest_exp=SimpleNamespace(exp_dir=tmp_dir))
            trainer.logger = MagicMock()
            handle = MagicMock()
            handle.url = "http://127.0.0.1:39563"
            cfg = SimpleNamespace(
                trace_config=TraceConfig(
                    enabled=True,
                    output_dir=None,
                    viewer_host="127.0.0.1",
                    viewer_port=39563,
                    viewer_refresh_interval_s=2.5,
                    viewer_scope="all",
                )
            )

            with (
                patch.object(rl_trainer, "get_rank", return_value=0),
                patch("xtuner.tools.producer_trace_viewer.start_trace_viewer", return_value=handle) as start_viewer,
            ):
                trainer._init_trace(cfg)
                trace_dir = Path(tmp_dir) / "producer_trace"
                self.assertTrue(trace_dir.exists())
                trainer._close_trace()

        start_viewer.assert_called_once_with(
            trace_dir,
            host="127.0.0.1",
            port=39563,
            refresh_interval_s=2.5,
            scope="all",
        )
        trainer.logger.info.assert_any_call("Producer Trace Viewer: http://127.0.0.1:39563")
        handle.close.assert_called_once()
