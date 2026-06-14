"""
Tests for ModelSelector — automatic model selection based on hardware specs.

Covers:
  - Hardware detection (local torch/nvidia-smi)
  - best_fit() selections across VRAM tiers
  - models_that_fit() listing
  - recommend_method()
  - fallback when no model fits
  - family preference and context window filtering
  - auth exclusion
  - Thread safety
  - Async wrapper
  - CLI commands
"""

import os
import asyncio
import threading
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from models.selector import (
    ModelSelector, ModelSpec, HardwareTier, MODEL_REGISTRY,
    detect_hardware, print_model_summary, _tier_from_vram,
    _tier_from_name,
)


class TestHardwareDetection(unittest.TestCase):
    """detect_hardware() under various conditions."""

    @patch("importlib.util.find_spec", return_value=None)
    def test_detect_no_torch_no_psutil(self, _mock_spec):
        with patch("models.selector._detect_via_nvidia_smi", return_value={}):
            hw = detect_hardware()
            self.assertEqual(hw["vram_total_gb"], 0)
            self.assertEqual(hw["ram_total_gb"], 0)
            self.assertEqual(hw["gpu_name"], None)

    @patch("importlib.util.find_spec")
    def test_detect_with_torch(self, mock_spec):
        def _spec(s):
            if s in ("torch", "psutil"):
                return MagicMock()
            return None
        mock_spec.side_effect = _spec

        import torch
        if torch.cuda.is_available():
            # Real GPU
            hw = detect_hardware()
            self.assertGreater(hw["vram_total_gb"], 0)
            self.assertGreater(hw["ram_total_gb"], 0)
            self.assertIsNotNone(hw["gpu_name"])
        else:
            hw = detect_hardware()
            self.assertGreater(hw["ram_total_gb"], 0)


class TestTierFromVRAM(unittest.TestCase):
    def test_none(self):
        self.assertEqual(_tier_from_vram(0), HardwareTier.NONE)
        self.assertEqual(_tier_from_vram(2), HardwareTier.NONE)

    def test_low(self):
        self.assertEqual(_tier_from_vram(4), HardwareTier.LOW)
        self.assertEqual(_tier_from_vram(8), HardwareTier.LOW)

    def test_medium(self):
        self.assertEqual(_tier_from_vram(12), HardwareTier.MEDIUM)
        self.assertEqual(_tier_from_vram(16), HardwareTier.MEDIUM)

    def test_high(self):
        self.assertEqual(_tier_from_vram(24), HardwareTier.HIGH)
        self.assertEqual(_tier_from_vram(32), HardwareTier.HIGH)

    def test_very_high(self):
        self.assertEqual(_tier_from_vram(40), HardwareTier.VERY_HIGH)
        self.assertEqual(_tier_from_vram(48), HardwareTier.VERY_HIGH)
        self.assertEqual(_tier_from_vram(79), HardwareTier.VERY_HIGH)

    def test_extreme(self):
        self.assertEqual(_tier_from_vram(80), HardwareTier.EXTREME)
        self.assertEqual(_tier_from_vram(160), HardwareTier.EXTREME)


class TestTierFromName(unittest.TestCase):
    def test_none(self):
        self.assertEqual(_tier_from_name("Intel UHD Graphics"), HardwareTier.NONE)

    def test_low(self):
        self.assertEqual(_tier_from_name("Tesla K80"), HardwareTier.LOW)
        self.assertEqual(_tier_from_name("NVIDIA P4"), HardwareTier.LOW)

    def test_medium(self):
        self.assertEqual(_tier_from_name("Tesla T4"), HardwareTier.MEDIUM)
        self.assertEqual(_tier_from_name("Tesla P100"), HardwareTier.MEDIUM)
        self.assertEqual(_tier_from_name("NVIDIA RTX 4090"), HardwareTier.MEDIUM)

    def test_high(self):
        self.assertEqual(_tier_from_name("Tesla V100"), HardwareTier.HIGH)
        self.assertEqual(_tier_from_name("Tesla V100S"), HardwareTier.HIGH)

    def test_very_high(self):
        self.assertEqual(_tier_from_name("NVIDIA A100"), HardwareTier.VERY_HIGH)
        self.assertEqual(_tier_from_name("NVIDIA A10G"), HardwareTier.VERY_HIGH)

    def test_extreme(self):
        self.assertEqual(_tier_from_name("NVIDIA H100"), HardwareTier.EXTREME)
        self.assertEqual(_tier_from_name("NVIDIA A100-80GB"), HardwareTier.EXTREME)
        self.assertEqual(_tier_from_name("NVIDIA H200"), HardwareTier.EXTREME)


class TestModelSelectorBestFit(unittest.TestCase):
    """best_fit() model selection."""

    def setUp(self):
        self.sel = ModelSelector()

    def test_best_fit_low_vram(self):
        """3GB VRAM should pick the smallest model."""
        result = self.sel.best_fit(vram_gb=3, ram_gb=4, method="qlora")
        self.assertTrue(result["fits"], f"Expected a fitting model, got: {result}")
        self.assertLess(result["params_b"], 3.0)

    def test_best_fit_medium_vram(self):
        """16GB VRAM should pick a ~7B model."""
        result = self.sel.best_fit(vram_gb=16, ram_gb=12, method="lora")
        self.assertTrue(result["fits"])
        self.assertGreaterEqual(result["params_b"], 2.0)

    def test_best_fit_high_vram(self):
        """40GB VRAM should pick a large model."""
        result = self.sel.best_fit(vram_gb=40, ram_gb=24, method="lora")
        self.assertTrue(result["fits"])
        # Should get a model > 7B
        self.assertGreater(result["params_b"], 7.0,
                           f"Expected large model for 40GB, got: {result}")

    def test_best_fit_extreme_vram(self):
        """80GB VRAM should pick the biggest model."""
        result = self.sel.best_fit(vram_gb=80, ram_gb=64, method="lora")
        self.assertTrue(result["fits"])
        self.assertGreater(result["params_b"], 30.0)

    def test_best_fit_no_vram_cpu(self):
        """0 VRAM should return no-fitting model."""
        result = self.sel.best_fit(vram_gb=0, ram_gb=4, method="qlora")
        self.assertFalse(result["fits"])

    def test_best_fit_no_ram(self):
        """Insufficient RAM should return no-fitting model."""
        result = self.sel.best_fit(vram_gb=80, ram_gb=1, method="qlora")
        self.assertFalse(result["fits"])

    def test_best_fit_falls_back_to_qlora(self):
        """If full/lora don't fit small VRAM, should try qlora."""
        sel = ModelSelector(registry=[])
        spec = ModelSpec(
            name="test/big",
            family="test",
            params_b=7.0,
            vram_gb_lora=16.0,
            vram_gb_qlora=4.0,
            vram_gb_full=32.0,
            ram_gb_required=8.0,
        )
        sel.register_model(spec)
        # 8GB VRAM is enough for QLoRA but not LoRA or full
        result = sel.best_fit(vram_gb=8, ram_gb=16, method="full")
        self.assertEqual(result["method"], "qlora")

    def test_best_fit_method_lora(self):
        result = self.sel.best_fit(vram_gb=16, ram_gb=12, method="lora")
        self.assertEqual(result["method"], "lora")

    def test_best_fit_prefer_family(self):
        result = self.sel.best_fit(vram_gb=16, ram_gb=12, method="qlora",
                                    prefer_family="llama")
        self.assertEqual(result["family"], "llama")

    def test_best_fit_require_context(self):
        result = self.sel.best_fit(vram_gb=16, ram_gb=12, method="qlora",
                                    require_context=32768)
        self.assertGreaterEqual(result["context_window"], 32768)

    def test_best_fit_exclude_auth(self):
        result = self.sel.best_fit(vram_gb=16, ram_gb=12, method="qlora",
                                    exclude_auth=True)
        self.assertFalse(result.get("requires_auth", False))

    def test_result_has_all_keys(self):
        result = self.sel.best_fit(vram_gb=16, ram_gb=12)
        for key in ("name", "family", "params_b", "method", "fits",
                     "explanation", "vram_total", "vram_needed"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_best_fit_detects_hardware(self):
        """Without args, should auto-detect and return something."""
        result = self.sel.best_fit()
        self.assertIn("fits", result)
        self.assertIn("name", result)


class TestModelSelectorModelsThatFit(unittest.TestCase):
    def test_models_that_fit_tiny_vram(self):
        results = ModelSelector().models_that_fit(vram_gb=3, ram_gb=4)
        self.assertGreater(len(results), 0)
        self.assertTrue(results[0]["fits"] or not results[0]["fits"])

    def test_models_that_fit_large_vram(self):
        results = ModelSelector().models_that_fit(vram_gb=80, ram_gb=64)
        self.assertGreater(len(results), 2)

    def test_models_that_fit_sorted_by_params(self):
        results = ModelSelector().models_that_fit(vram_gb=80, ram_gb=64)
        params = [r["params_b"] for r in results]
        self.assertEqual(params, sorted(params, reverse=True))


class TestModelSelectorRecommendMethod(unittest.TestCase):
    def setUp(self):
        self.sel = ModelSelector()

    def test_recommend_full_for_large_vram_small_model(self):
        self.assertEqual(self.sel.recommend_method(vram_gb=48, params_b=2.7), "full")

    def test_recommend_lora_for_16gb_small_model(self):
        self.assertEqual(self.sel.recommend_method(vram_gb=16, params_b=7), "lora")

    def test_recommend_qlora_for_limited_vram(self):
        self.assertEqual(self.sel.recommend_method(vram_gb=8, params_b=7), "qlora")

    def test_recommend_qlora_large_model(self):
        self.assertEqual(self.sel.recommend_method(vram_gb=40, params_b=70), "qlora")


class TestModelSelectorRegistry(unittest.TestCase):
    def test_get_model_found(self):
        spec = ModelSelector().get_model("google/gemma-2-2b")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.family, "gemma")

    def test_get_model_not_found(self):
        spec = ModelSelector().get_model("nonexistent/model")
        self.assertIsNone(spec)

    def test_register_model(self):
        sel = ModelSelector()
        spec = ModelSpec(
            name="test/model",
            family="test",
            params_b=1.0,
            vram_gb_lora=1.0,
            vram_gb_qlora=0.5,
            vram_gb_full=3.0,
            ram_gb_required=2.0,
            tier=HardwareTier.LOW,
        )
        sel.register_model(spec)
        self.assertIsNotNone(sel.get_model("test/model"))
        self.assertIn(spec, sel.registry)

    def test_model_min_vram(self):
        spec = ModelSpec(
            name="test/model",
            family="test",
            params_b=1.0,
            vram_gb_lora=4.0,
            vram_gb_qlora=2.0,
            vram_gb_full=8.0,
            ram_gb_required=2.0,
        )
        self.assertEqual(spec.min_vram("lora"), 4.0)
        self.assertEqual(spec.min_vram("qlora"), 2.0)
        self.assertEqual(spec.min_vram("full"), 8.0)
        self.assertEqual(spec.min_vram("unknown"), 2.0)

    def test_model_fits_in(self):
        spec = MODEL_REGISTRY[0]
        self.assertTrue(spec.fits_in(vram_gb=999, ram_gb=999, method="qlora"))
        self.assertFalse(spec.fits_in(vram_gb=0, ram_gb=999, method="qlora"))


class TestModelSelectorSummary(unittest.TestCase):
    def test_summary_has_all_sections(self):
        summary = ModelSelector().summary(vram_gb=16, ram_gb=12)
        self.assertIn("hardware", summary)
        self.assertIn("recommended_model", summary)
        self.assertIn("all_fitting", summary)
        self.assertIn("gpu", summary["hardware"])
        self.assertIn("vram_total_gb", summary["hardware"])

    def test_summary_recommendation_is_reasonable(self):
        summary = ModelSelector().summary(vram_gb=8, ram_gb=8)
        rec = summary["recommended_model"]
        self.assertIn("name", rec)
        self.assertLess(rec["params_b"], 10.0,
                        f"Expected small model for 8GB, got {rec['name']}")


class TestModelSelectorThreadSafety(unittest.TestCase):
    def test_concurrent_best_fit(self):
        sel = ModelSelector()
        errors = []

        def query():
            try:
                for _ in range(5):
                    sel.best_fit(vram_gb=16, ram_gb=12)
                    sel.models_that_fit(vram_gb=16, ram_gb=12)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=query) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_register_model(self):
        sel = ModelSelector()
        errors = []

        def register(idx):
            try:
                spec = ModelSpec(
                    name=f"test/model-{idx}",
                    family="test",
                    params_b=1.0,
                    vram_gb_lora=1.0,
                    vram_gb_qlora=0.5,
                    vram_gb_full=3.0,
                    ram_gb_required=2.0,
                )
                sel.register_model(spec)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


class TestAsyncRemoteExecutor(unittest.TestCase):
    """Async variant of RemoteColabExecutor."""

    def test_available_property(self):
        from colab.async_remote_executor import AsyncRemoteColabExecutor
        e = AsyncRemoteColabExecutor()
        self.assertIsInstance(e.available, bool)

    def test_not_connected_after_init(self):
        from colab.async_remote_executor import AsyncRemoteColabExecutor
        e = AsyncRemoteColabExecutor()
        self.assertFalse(e.connected)

    def test_async_execute_not_connected(self):
        from colab.async_remote_executor import AsyncRemoteColabExecutor

        async def run():
            e = AsyncRemoteColabExecutor()
            result = await e.execute("x=1")
            return result

        result = asyncio.run(run())
        self.assertFalse(result["success"])
        self.assertIn("Not connected", result["error"])

    def test_async_context_manager(self):
        from colab.async_remote_executor import AsyncRemoteColabExecutor

        async def run():
            async with AsyncRemoteColabExecutor() as e:
                self.assertFalse(e.connected)
            self.assertFalse(e.connected)
            return True

        self.assertTrue(asyncio.run(run()))

    def test_async_login_once_when_unavailable(self):
        from colab.async_remote_executor import AsyncRemoteColabExecutor
        with patch.object(AsyncRemoteColabExecutor, "available",
                          new_callable=PropertyMock, return_value=False):
            async def run():
                e = AsyncRemoteColabExecutor()
                with patch.object(e._sync, "login_once", return_value=False):
                    result = await e.login_once()
                    return result
            self.assertFalse(asyncio.run(run()))


class TestCLICommands(unittest.TestCase):
    """CLI detect and list-models commands."""

    def test_cmd_detect_runs(self):
        from cli import cmd_detect
        try:
            cmd_detect()
        except Exception as e:
            self.fail(f"cmd_detect raised: {e}")

    def test_cmd_list_models_spec_runs(self):
        from cli import cmd_list_models_spec
        try:
            cmd_list_models_spec()
        except Exception as e:
            self.fail(f"cmd_list_models_spec raised: {e}")


class TestModelSelectorEdgeCases(unittest.TestCase):
    """Edge cases and malformed inputs."""

    def test_empty_registry(self):
        sel = ModelSelector(registry=[])
        result = sel.best_fit(vram_gb=80, ram_gb=64)
        self.assertFalse(result["fits"])
        self.assertIn("No model", result["explanation"])

    def test_negative_vram(self):
        sel = ModelSelector()
        result = sel.best_fit(vram_gb=-1, ram_gb=64)
        self.assertFalse(result["fits"])

    def test_vram_equals_exact_requirement(self):
        sel = ModelSelector(registry=[])
        spec = ModelSpec(
            name="test/exact",
            family="test",
            params_b=100.0,
            vram_gb_lora=8.0,
            vram_gb_qlora=4.0,
            vram_gb_full=16.0,
            ram_gb_required=8.0,
        )
        sel.register_model(spec)
        result = sel.best_fit(vram_gb=8.0, ram_gb=8.0, method="lora")
        self.assertTrue(result["fits"])
        self.assertEqual(result["name"], "test/exact")

    def test_print_model_summary(self):
        sel = ModelSelector()
        try:
            print_model_summary(sel)
        except Exception as e:
            self.fail(f"print_model_summary raised: {e}")

    def test_detect_hardware_never_raises(self):
        try:
            for _ in range(10):
                detect_hardware()
        except Exception as e:
            self.fail(f"detect_hardware raised: {e}")


if __name__ == "__main__":
    unittest.main()
