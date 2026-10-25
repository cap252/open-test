"""Process-level contracts for shared caches and exclusive execution directories."""

import multiprocessing as mp
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch


def _cache_worker(pipe, directory, operation, **options):
    import fcntl

    flock = fcntl.flock

    def observed_flock(stream, mode):
        if mode in (fcntl.LOCK_SH, fcntl.LOCK_EX):
            try:
                return flock(stream, mode | fcntl.LOCK_NB)
            except BlockingIOError:
                pipe.send(("blocked", operation))
        return flock(stream, mode)

    with patch.object(fcntl, "flock", observed_flock):
        _cache_operation(pipe, directory, operation, **options)


def _cache_operation(
    pipe, directory, operation, *, hold=None, variable=False, fail=False
):
    """Hold a real cache operation at an IPC barrier, without scheduler sleeps."""
    released = False

    def barrier(stage):
        nonlocal released
        if stage == hold and not released:
            released = True
            pipe.send(("held", stage))
            if pipe.recv() != "release":
                raise AssertionError("Unexpected barrier command")
            if fail:
                raise ValueError("Injected creator failure")

    pipe.send(("started", operation))
    try:
        if operation == "lock":
            from opennpu_quant._locking import cache_lock

            pipe.send(("attempt", operation))
            with cache_lock(directory, shared=variable):
                barrier("lock")
            result = None
        elif operation == "input":
            import numpy as np
            from opennpu_quant.data.feed_cache import prepare_feed_cache
            from opennpu_quant.data.variable_cache import prepare_variable_feed_cache

            made = False

            def factory():
                nonlocal made
                made = True
                barrier("factory")
                for size in (2, 3) if variable else (2, 2):
                    yield {"x": np.arange(size, dtype=np.float32).reshape(1, size)}

            def verify():
                barrier("source")

            prepare = prepare_variable_feed_cache if variable else prepare_feed_cache
            pipe.send(("attempt", operation))
            cache = prepare(
                directory,
                factory,
                identity={"source": "fixture"},
                verify_source=verify,
                expected_samples=2,
                max_bytes=1024**2,
            )
            result = dict(
                reused=cache.reused,
                factory_called=made,
                identity=cache.manifest["samples_identity"],
                values=[
                    (list(feed["x"].shape), feed["x"].tobytes().hex())
                    for feed in cache()
                ],
            )
        else:
            from opennpu_quant import CalibrationConfig, OrtConfig
            from opennpu_quant.quant import storage
            from opennpu_quant.quant.calibration import load_statistics
            from opennpu_quant._io import read_json
            from tests.library_fixtures import toy_model

            if operation == "read":
                validate = storage.validate_statistics

                def checked(*args, **kwargs):
                    validate(*args, **kwargs)
                    barrier("validation")

                pipe.send(("attempt", operation))
                with patch.object(storage, "validate_statistics", checked):
                    stats = storage.read_statistics(
                        directory,
                        toy_model(),
                        config=CalibrationConfig(),
                        ort=OrtConfig.cpu(),
                    )
            elif operation == "save":
                # Read a separate, immutable source before publishing to the target.
                source = Path(directory).with_name("statistics-source")
                stats = load_statistics(
                    source / read_json(source / "complete.json")["generation"]
                )
                write = storage.atomic_json

                def publish(path, value):
                    if Path(path).name == "complete.json":
                        barrier("publication")
                    write(path, value)

                pipe.send(("attempt", operation))
                with patch.object(storage, "atomic_json", publish):
                    storage.save_statistics(stats, directory)
            else:
                raise AssertionError(operation)
            result = stats.statistics_identity
        pipe.send(("done", result))
    except BaseException as error:
        pipe.send(("error", type(error).__name__, str(error)))
    finally:
        pipe.close()


class CacheConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.context = mp.get_context("spawn")

    def start(self, directory, operation, **options):
        parent, child = self.context.Pipe()
        process = self.context.Process(
            target=_cache_worker,
            args=(child, str(directory), operation),
            kwargs=options,
        )
        process.start()
        child.close()

        def stop():
            if process.is_alive():
                process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            parent.close()

        self.addCleanup(stop)
        self.assertEqual(("started", operation), self.receive(parent))
        self.assertEqual(("attempt", operation), self.receive(parent))
        return process, parent

    def receive(self, pipe):
        self.assertTrue(pipe.poll(20), "Cache operation did not reach its IPC barrier")
        return pipe.recv()

    def done(self, pipe):
        message = self.receive(pipe)
        self.assertEqual("done", message[0], message)
        return message[1]

    def seed_input(self, variable):
        directory = self.root / ("variable" if variable else "fixed")
        _, pipe = self.start(directory, "input", variable=variable)
        return directory, self.done(pipe)

    def seed_statistics(self):
        from opennpu_quant import calibrate
        from opennpu_quant.quant.storage import save_statistics
        from tests.library_fixtures import feeds, toy_model

        stats = calibrate(toy_model(), feeds)
        directory = self.root / "statistics"
        save_statistics(stats, directory)
        save_statistics(stats, self.root / "statistics-source")
        return directory, stats.statistics_identity

    def test_completed_fixed_and_variable_inputs_have_concurrent_readers(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                directory, expected = self.seed_input(variable)
                _, first = self.start(
                    directory, "input", variable=variable, hold="source"
                )
                self.assertEqual(("held", "source"), self.receive(first))
                _, second = self.start(directory, "input", variable=variable)
                other = self.done(second)
                self.assertTrue(other["reused"])
                self.assertFalse(other["factory_called"])
                self.assertEqual(expected["values"], other["values"])
                self.assertEqual(expected["identity"], other["identity"])
                first.send("release")
                self.assertEqual(other, self.done(first))

    def test_same_input_is_built_once_and_waiter_reuses_it(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                directory = self.root / str(variable)
                _, first = self.start(
                    directory, "input", variable=variable, hold="factory"
                )
                self.assertEqual(("held", "factory"), self.receive(first))
                _, second = self.start(directory, "input", variable=variable)
                self.assertEqual(("blocked", "input"), self.receive(second))
                first.send("release")
                created, reused = self.done(first), self.done(second)
                self.assertTrue(created["factory_called"])
                self.assertFalse(created["reused"])
                self.assertTrue(reused["reused"])
                self.assertFalse(reused["factory_called"])
                self.assertEqual(created["values"], reused["values"])
                self.assertEqual(created["identity"], reused["identity"])

    def test_failed_creator_releases_waiter_without_publishing_partial_cache(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                directory = self.root / str(variable)
                _, first = self.start(
                    directory, "input", variable=variable, hold="factory", fail=True
                )
                self.assertEqual(("held", "factory"), self.receive(first))
                _, second = self.start(directory, "input", variable=variable)
                self.assertEqual(("blocked", "input"), self.receive(second))
                first.send("release")
                self.assertEqual("ValueError", self.receive(first)[1])
                recovered = self.done(second)
                self.assertTrue(recovered["factory_called"])
                self.assertFalse(recovered["reused"])
                self.assertFalse(list(self.root.glob("*.building-*")))

    def test_statistics_readers_overlap_and_preserve_payloads(self):
        directory, expected = self.seed_statistics()
        before = {
            p.relative_to(directory): p.read_bytes()
            for p in directory.rglob("*")
            if p.is_file()
        }
        _, first = self.start(directory, "read", hold="validation")
        self.assertEqual(("held", "validation"), self.receive(first))
        _, second = self.start(directory, "read")
        self.assertEqual(expected, self.done(second))
        first.send("release")
        self.assertEqual(expected, self.done(first))
        self.assertEqual(
            before,
            {
                p.relative_to(directory): p.read_bytes()
                for p in directory.rglob("*")
                if p.is_file()
            },
        )

    def test_statistics_writer_waits_for_active_readers(self):
        directory, expected = self.seed_statistics()
        _, first = self.start(directory, "read", hold="validation")
        self.assertEqual(("held", "validation"), self.receive(first))
        _, writer = self.start(directory, "save", hold="publication")
        self.assertEqual(("blocked", "save"), self.receive(writer))
        first.send("release")
        self.assertEqual(expected, self.done(first))
        self.assertEqual(("held", "publication"), self.receive(writer))
        writer.send("release")
        self.assertEqual(expected, self.done(writer))

    def test_statistics_reader_waits_for_publication_and_then_validates(self):
        directory, expected = self.seed_statistics()
        _, writer = self.start(directory, "save", hold="publication")
        self.assertEqual(("held", "publication"), self.receive(writer))
        _, reader = self.start(directory, "read")
        self.assertEqual(("blocked", "read"), self.receive(reader))
        writer.send("release")
        self.assertEqual(expected, self.done(writer))
        self.assertEqual(expected, self.done(reader))

    def test_waiting_statistics_reader_still_rejects_tampering(self):
        directory, _ = self.seed_statistics()
        _, writer = self.start(directory, "save", hold="publication")
        self.assertEqual(("held", "publication"), self.receive(writer))
        _, reader = self.start(directory, "read")
        self.assertEqual(("blocked", "read"), self.receive(reader))
        payload = next(directory.glob("generations/*/histograms.npz"))
        payload.write_bytes(payload.read_bytes() + b"tampered")
        writer.send("release")
        self.done(writer)
        error = self.receive(reader)
        self.assertEqual(("error", "ValueError"), error[:2])
        self.assertIn("modified", error[2])

    def test_process_exit_releases_writer_for_waiting_reader(self):
        path = self.root / "cache.lock"
        owner, first = self.start(path, "lock", hold="lock")
        self.assertEqual(("held", "lock"), self.receive(first))
        _, second = self.start(path, "lock", variable=True)
        self.assertEqual(("blocked", "lock"), self.receive(second))
        owner.terminate()
        owner.join(5)
        self.assertFalse(owner.is_alive())
        self.assertIsNone(self.done(second))

    def test_wait_is_interruptible_and_does_not_release_the_owner(self):
        import os
        from opennpu_quant._locking import RunLock

        path = self.root / "cache.lock"
        _, first = self.start(path, "lock", hold="lock")
        self.assertEqual(("held", "lock"), self.receive(first))
        waiter, second = self.start(path, "lock")
        self.assertEqual(("blocked", "lock"), self.receive(second))
        os.kill(waiter.pid, signal.SIGINT)
        self.assertEqual("KeyboardInterrupt", self.receive(second)[1])
        with self.assertRaisesRegex(RuntimeError, "Another process"):
            with RunLock(path):
                self.fail("Interrupted waiter released another process's lock")
        first.send("release")
        self.done(first)
        with RunLock(path):
            pass

    def test_unrelated_cache_path_and_execution_lock_policy_are_independent(self):
        from opennpu_quant._locking import RunLock

        path = self.root / "run.lock"
        with RunLock(path):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with RunLock(path):
                    self.fail("Concurrent run accepted")
            _, other = self.start(self.root / "other.lock", "lock")
            self.assertIsNone(self.done(other))
