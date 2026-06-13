"""Verify all 13 capability modules generate valid Python code."""
import sys
import os
import ast
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _check_code(code: str, name: str):
    """Verify generated code is valid Python (after stripping shell cmds)."""
    assert code and isinstance(code, str), f"{name}: no code returned"
    lines = [ln for ln in code.split("\n") if not ln.strip().startswith("!")]
    py_code = "\n".join(lines)
    try:
        ast.parse(py_code if py_code.strip() else "pass")
    except SyntaxError as e:
        raise AssertionError(f"{name}: SyntaxError: {e}\nCode:\n{code[:300]}")


def test_data_collectors():
    from capabilities.data_collectors import (
        YouTubeAudioDownloader, TwitterScraper, PDFKnowledgeExtractor,
        GitHubReadmeCollector, WikipediaDumpStreamer, SlackExportParser,
        ArxivPaperFetcher, RedditAPICollector, CommonCrawlExtractor,
        SeleniumDynamicScraper
    )
    c = YouTubeAudioDownloader()
    _check_code(c.download_code("https://youtube.com/watch?v=test"), "YouTubeAudioDownloader.download_code")
    c = TwitterScraper()
    _check_code(c.scrape_code("ml", limit=5), "TwitterScraper.scrape_code")
    c = PDFKnowledgeExtractor()
    _check_code(c.parse_code("test.pdf"), "PDFKnowledgeExtractor.parse_code")
    c = GitHubReadmeCollector()
    _check_code(c.collect_code("transformers"), "GitHubReadmeCollector.collect_code")
    c = WikipediaDumpStreamer()
    _check_code(c.stream_code(), "WikipediaDumpStreamer.stream_code")
    c = SlackExportParser()
    _check_code(c.parse_code("test.zip"), "SlackExportParser.parse_code")
    c = ArxivPaperFetcher()
    _check_code(c.fetch_code(max_results=5), "ArxivPaperFetcher.fetch_code")
    c = RedditAPICollector()
    _check_code(c.collect_code("machinelearning", limit=5), "RedditAPICollector.collect_code")
    c = CommonCrawlExtractor()
    _check_code(c.extract_code(max_pages=5), "CommonCrawlExtractor.extract_code")
    c = SeleniumDynamicScraper()
    _check_code(c.scrape_code("https://example.com"), "SeleniumDynamicScraper.scrape_code")
    print("  PASS: 10 data_collectors")


def test_data_processors():
    from capabilities.data_processors import (
        SyntheticDataGenerator, DataAugmenter, PrivacyMasker,
        ToxicityFilter, DeduplicationEngine
    )
    c = SyntheticDataGenerator()
    _check_code(c.generate_code(task_types=["classification"], num_per_task=5), "SyntheticDataGenerator.generate_code")
    c = DataAugmenter()
    _check_code(c.augment_code(methods=["synonym"]), "DataAugmenter.augment_code")
    c = PrivacyMasker()
    _check_code(c.mask_code(), "PrivacyMasker.mask_code")
    c = ToxicityFilter()
    _check_code(c.filter_code(), "ToxicityFilter.filter_code")
    c = DeduplicationEngine()
    _check_code(c.dedup_code(), "DeduplicationEngine.dedup_code")
    print("  PASS: 5 data_processors")


def test_executors():
    from capabilities.executors import MultimodalDataLoader, DockerSandbox, ArtifactUploader
    c = MultimodalDataLoader()
    _check_code(c.load_code(), "MultimodalDataLoader.load_code")
    c = DockerSandbox(".")
    _check_code(c.setup_code(), "DockerSandbox.setup_code")
    c = ArtifactUploader()
    _check_code(c.upload_code(destination="huggingface"), "ArtifactUploader.upload_code")
    print("  PASS: 3 executors")


def test_experiment_tracking():
    from capabilities.experiment_tracking import (
        ExperimentTracker, HyperparameterDatabase, OptimalConfigRecommender,
        DriftDetector, AblationStudyRunner
    )
    c = ExperimentTracker()
    _check_code(c.setup_code(), "ExperimentTracker.setup_code")
    _check_code(c.log_metrics_code(), "ExperimentTracker.log_metrics_code")
    _check_code(c.finish_code(), "ExperimentTracker.finish_code")
    c = HyperparameterDatabase()
    _check_code(c.init_code(), "HyperparameterDatabase.init_code")
    _check_code(c.log_trial_code(), "HyperparameterDatabase.log_trial_code")
    _check_code(c.best_config_code(), "HyperparameterDatabase.best_config_code")
    c = OptimalConfigRecommender()
    _check_code(c.recommend_code(model_params_b=1.5, dataset_size=10000), "OptimalConfigRecommender.recommend_code")
    c = DriftDetector()
    _check_code(c.detect_code(), "DriftDetector.detect_code")
    c = AblationStudyRunner()
    _check_code(c.run_code(["lora", "qlora"]), "AblationStudyRunner.run_code")
    print("  PASS: 9 experiment_tracking")


def test_model_optimization():
    from capabilities.model_optimization import (
        LoRAAdapterZoo, ModelMerger, GradientCheckpointer,
        MixedPrecisionSelector, FlashAttentionIntegration, QuantizationAwareTraining
    )
    c = LoRAAdapterZoo()
    _check_code(c.save_adapter_code("sentiment", "microsoft/phi-2"), "LoRAAdapterZoo.save_adapter_code")
    _check_code(c.switch_adapter_code(), "LoRAAdapterZoo.switch_adapter_code")
    _check_code(c.list_adapters_code(), "LoRAAdapterZoo.list_adapters_code")
    c = ModelMerger()
    _check_code(c.merge_code(), "ModelMerger.merge_code")
    _check_code(c.ties_merge_code([{"path": "./a", "weight": 1.0}]), "ModelMerger.ties_merge_code")
    _check_code(GradientCheckpointer.enable_code(), "GradientCheckpointer.enable_code")
    _check_code(MixedPrecisionSelector.detect_code(), "MixedPrecisionSelector.detect_code")
    _check_code(MixedPrecisionSelector.training_args_code(), "MixedPrecisionSelector.training_args_code")
    _check_code(FlashAttentionIntegration.install_code(), "FlashAttentionIntegration.install_code")
    _check_code(QuantizationAwareTraining.setup_code(), "QuantizationAwareTraining.setup_code")
    print("  PASS: 10 model_optimization")


def test_training_utils():
    from capabilities.training_utils import (
        EarlyStoppingCallback, LearningRateScheduler, DistributedTraining,
        CheckpointManager, BestModelSelector, EvaluationPipeline,
        DatasetSplitter, DataCollatorGenerator
    )
    c = EarlyStoppingCallback()
    _check_code(c.huggingface_code(), "EarlyStoppingCallback.huggingface_code")
    _check_code(LearningRateScheduler.setup_code(), "LearningRateScheduler.setup_code")
    _check_code(DistributedTraining.deepspeed_config_code(), "DistributedTraining.deepspeed_config_code")
    c = CheckpointManager()
    _check_code(c.save_code(), "CheckpointManager.save_code")
    c = BestModelSelector()
    _check_code(c.huggingface_code(), "BestModelSelector.huggingface_code")
    _check_code(EvaluationPipeline.compute_code(), "EvaluationPipeline.compute_code")
    c = DatasetSplitter()
    _check_code(c.split_code(), "DatasetSplitter.split_code")
    _check_code(DataCollatorGenerator.select_code(), "DataCollatorGenerator.select_code")
    print("  PASS: 8 training_utils")


def test_runtime():
    from capabilities.runtime import (
        GPUDetector, RuntimeSwitcher, SessionKeepAlive,
        DisconnectionHandler, ColabLimitsDetector, RuntimeRecommendationEngine,
        MultiGPUChecker, ResourceReleaseHandler, RuntimeBenchmark
    )
    _check_code(GPUDetector.detect_code(), "GPUDetector.detect_code")
    _check_code(RuntimeSwitcher.switch_code(), "RuntimeSwitcher.switch_code")
    _check_code(SessionKeepAlive.enable_code(), "SessionKeepAlive.enable_code")
    c = DisconnectionHandler(".")
    _check_code(c.save_state_code(), "DisconnectionHandler.save_state_code")
    _check_code(ColabLimitsDetector.detect_code(), "ColabLimitsDetector.detect_code")
    _check_code(RuntimeRecommendationEngine.recommend_code(), "RuntimeRecommendationEngine.recommend_code")
    _check_code(MultiGPUChecker.detect_code(), "MultiGPUChecker.detect_code")
    _check_code(ResourceReleaseHandler.cleanup_code(), "ResourceReleaseHandler.cleanup_code")
    _check_code(RuntimeBenchmark.benchmark_code(), "RuntimeBenchmark.benchmark_code")
    print("  PASS: 9 runtime")


def test_github_advanced():
    from capabilities.github_advanced import (
        GitHubRepoManager, ErrorCommitAutomation, SimilarErrorRetriever,
        SolutionLearner, GitHubPullRequester, ConversationArchiver,
        MetricsVisualizerGitHub, VersionTagOnSuccess
    )
    c = GitHubRepoManager(".")
    _check_code(c.ensure_repo_code("test-repo"), "GitHubRepoManager.ensure_repo_code")
    _check_code(c.commit_code(), "GitHubRepoManager.commit_code")
    c = ErrorCommitAutomation(".")
    _check_code(c.capture_and_commit_code("job1", "print('hi')", "error", [], "T4"), "ErrorCommitAutomation.capture_and_commit_code")
    c = SimilarErrorRetriever(".")
    _check_code(c.retrieve_code("CUDA OOM"), "SimilarErrorRetriever.retrieve_code")
    c = SolutionLearner(".")
    _check_code(c.record_solution_code("oom", "CUDA out of memory", ["clear cache"], "del model"), "SolutionLearner.record_solution_code")
    c = GitHubPullRequester()
    _check_code(c.create_pr_code("user/repo", "job1"), "GitHubPullRequester.create_pr_code")
    c = ConversationArchiver(".")
    _check_code(c.archive_code("job1", [{"role": "user", "content": "hi"}]), "ConversationArchiver.archive_code")
    c = MetricsVisualizerGitHub(".")
    _check_code(c.generate_report_code("job1", {"loss": 0.5}), "MetricsVisualizerGitHub.generate_report_code")
    c = VersionTagOnSuccess()
    _check_code(c.tag_code("job1", "model", {"acc": 0.9}), "VersionTagOnSuccess.tag_code")
    print("  PASS: 9 github_advanced")


def test_storage_db():
    from capabilities.storage_db import (
        SQLiteJobStore, MetricsTimeSeriesDB, ModelRegistry,
        ArtifactCompressor, StorageCleanupPolicy, ExportToHTML, CrossSessionMemory
    )
    c = SQLiteJobStore(":memory:")
    _check_code(c.init_code(), "SQLiteJobStore.init_code")
    _check_code(c.crud_code(), "SQLiteJobStore.crud_code")
    c = MetricsTimeSeriesDB(":memory:")
    _check_code(c.init_code(), "MetricsTimeSeriesDB.init_code")
    _check_code(c.insert_and_query_code(), "MetricsTimeSeriesDB.insert_and_query_code")
    c = ModelRegistry(":memory:")
    _check_code(c.registry_code(), "ModelRegistry.registry_code")
    c = ArtifactCompressor(".")
    _check_code(c.compress_code("job1", ["model.safetensors"]), "ArtifactCompressor.compress_code")
    c = StorageCleanupPolicy()
    _check_code(c.cleanup_code(), "StorageCleanupPolicy.cleanup_code")
    c = ExportToHTML(".")
    _check_code(c.export_code("job1", "model", "dataset", "T4", "ok", {"loss": 0.5}), "ExportToHTML.export_code")
    c = CrossSessionMemory(":memory:")
    _check_code(c.memory_code(), "CrossSessionMemory.memory_code")
    print("  PASS: 9 storage_db")


def test_ui_builder():
    from capabilities.ui_builder import (
        GradioChatInterface, StreamlitDashboard, IPythonWidgetController,
        TerminalMode, HumanApprovalModal, CodeEditorWithDiff
    )
    _check_code(GradioChatInterface.build_code(), "GradioChatInterface.build_code")
    _check_code(StreamlitDashboard.build_code(), "StreamlitDashboard.build_code")
    _check_code(IPythonWidgetController.build_code(), "IPythonWidgetController.build_code")
    _check_code(TerminalMode.cli_code(), "TerminalMode.cli_code")
    _check_code(HumanApprovalModal.modal_code(), "HumanApprovalModal.modal_code")
    _check_code(CodeEditorWithDiff.diff_code(), "CodeEditorWithDiff.diff_code")
    print("  PASS: 6 ui_builder")


def test_security():
    from capabilities.security import (
        CommandBlocklist, NetworkTrafficLogger, ResourceQuotaEnforcer,
        DataLeakDetector, UserConfirmationOnExternalPush,
        SessionEncryption, AnomalyDetector
    )
    _check_code(CommandBlocklist.check_code(), "CommandBlocklist.check_code")
    c = NetworkTrafficLogger(".")
    _check_code(c.logging_code(), "NetworkTrafficLogger.logging_code")
    c = ResourceQuotaEnforcer()
    _check_code(c.enforce_code(), "ResourceQuotaEnforcer.enforce_code")
    _check_code(DataLeakDetector.scan_code(), "DataLeakDetector.scan_code")
    _check_code(UserConfirmationOnExternalPush.confirmation_code(), "UserConfirmationOnExternalPush.confirmation_code")
    _check_code(SessionEncryption.encrypt_code(), "SessionEncryption.encrypt_code")
    c = AnomalyDetector()
    _check_code(c.detect_code(), "AnomalyDetector.detect_code")
    print("  PASS: 7 security")


def test_advanced_training():
    from capabilities.advanced_training import (
        HyperparameterOptimizer, MultiModelExperimentRunner,
        DatasetSynthesizer, PromptOptimizer, ChainOfThoughtTracing,
        ModelComparisonReport
    )
    c = HyperparameterOptimizer()
    _check_code(c.optimize_code(n_trials=5), "HyperparameterOptimizer.optimize_code")
    c = MultiModelExperimentRunner()
    _check_code(c.run_code(["microsoft/phi-2"], "test"), "MultiModelExperimentRunner.run_code")
    c = DatasetSynthesizer()
    _check_code(c.synthesize_code(num_examples=5), "DatasetSynthesizer.synthesize_code")
    c = PromptOptimizer()
    _check_code(c.optimize_code("Classify", ["text"], ["pos"]), "PromptOptimizer.optimize_code")
    c = ChainOfThoughtTracing()
    _check_code(c.tracing_code(), "ChainOfThoughtTracing.tracing_code")
    c = ModelComparisonReport()
    _check_code(c.generate_code([{"model": "a", "acc": 0.9}]), "ModelComparisonReport.generate_code")
    print("  PASS: 6 advanced_training")


def test_autonomous_intelligence():
    from capabilities.autonomous_intelligence import (
        AutomatedPaperReproducer, SelfHealingAgent,
        LearningFromFeedback, MetaAgent
    )
    c = AutomatedPaperReproducer(".")
    _check_code(c.reproduce_code("2301.00234"), "AutomatedPaperReproducer.reproduce_code")
    c = SelfHealingAgent()
    _check_code(c.healing_code(), "SelfHealingAgent.healing_code")
    c = LearningFromFeedback()
    _check_code(c.feedback_code(), "LearningFromFeedback.feedback_code")
    c = MetaAgent()
    _check_code(c.meta_agent_code(), "MetaAgent.meta_agent_code")
    print("  PASS: 4 autonomous_intelligence")


if __name__ == "__main__":
    tests = [fn for fn in dir() if fn.startswith("test_")]
    passed = 0
    failed = 0
    for name in sorted(tests):
        fn = globals()[name]
        print(f"\n{name}...")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
