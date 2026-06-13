"""
Plugin system for Colab Agent.

Provides:
  - Plugin base class with lifecycle hooks
  - PluginRegistry for discovery and loading (thread-safe)
  - Decorator for easy plugin registration
  - Hook execution in OrchestratorAgent lifecycle
"""

import logging
import inspect
import threading
from typing import Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


HOOK_BEFORE_PLAN = "before_plan"
HOOK_AFTER_PLAN = "after_plan"
HOOK_BEFORE_STEP = "before_step"
HOOK_AFTER_STEP = "after_step"
HOOK_BEFORE_CODE_GEN = "before_code_gen"
HOOK_AFTER_CODE_GEN = "after_code_gen"
HOOK_ON_ERROR = "on_error"
HOOK_ON_SUMMARY = "on_summary"
HOOK_ON_COMPLETE = "on_complete"

ALL_HOOKS = frozenset({
    HOOK_BEFORE_PLAN, HOOK_AFTER_PLAN, HOOK_BEFORE_STEP, HOOK_AFTER_STEP,
    HOOK_BEFORE_CODE_GEN, HOOK_AFTER_CODE_GEN, HOOK_ON_ERROR,
    HOOK_ON_SUMMARY, HOOK_ON_COMPLETE,
})


class Plugin:
    """Base class for all plugins. Subclass and override hook methods."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""

    def __init__(self):
        if not self.name:
            self.name = self.__class__.__name__

    def before_plan(self, goal: str, context: dict) -> tuple[str, dict]:
        return goal, context

    def after_plan(self, plan: dict, context: dict) -> tuple[dict, dict]:
        return plan, context

    def before_step(self, step: dict, context: dict) -> tuple[dict, dict]:
        return step, context

    def after_step(self, step: dict, result: dict, context: dict) -> tuple[dict, dict]:
        return result, context

    def before_code_gen(self, step: dict, prompt: str, context: dict) -> tuple[str, dict]:
        return prompt, context

    def after_code_gen(self, step: dict, code: str, context: dict) -> tuple[str, dict]:
        return code, context

    def on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        return None, context

    def on_summary(self, summary: str, context: dict) -> tuple[str, dict]:
        return summary, context

    def on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        return result, context


@dataclass
class PluginMeta:
    cls: type
    instance: Optional[Plugin] = None
    enabled: bool = True
    config: dict = field(default_factory=dict)
    priority: int = 100


class PluginRegistry:
    """Thread-safe central registry for plugins."""

    def __init__(self):
        self._lock = threading.Lock()
        self._plugins: dict[str, PluginMeta] = {}

    def register(self, plugin_cls: type, config: dict = None,
                 enabled: bool = True, priority: int = 100) -> str:
        if not inspect.isclass(plugin_cls) or not issubclass(plugin_cls, Plugin):
            raise TypeError(f"{plugin_cls.__name__} must be a Plugin subclass")
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dict or None")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if not isinstance(priority, int):
            raise TypeError("priority must be an int")

        name = plugin_cls.name or plugin_cls.__name__
        with self._lock:
            if name in self._plugins:
                logger.warning(f"Plugin '{name}' already registered, overwriting")
            meta = PluginMeta(
                cls=plugin_cls,
                config=config or {},
                enabled=enabled,
                priority=priority,
            )
            self._plugins[name] = meta
        logger.info(f"Registered plugin: {name} (v{plugin_cls.version})")
        return name

    def unregister(self, name: str):
        with self._lock:
            self._plugins.pop(name, None)

    def get(self, name: str) -> Optional[Plugin]:
        with self._lock:
            meta = self._plugins.get(name)
            if meta is None or not meta.enabled:
                return None
            if meta.instance is None:
                meta.instance = meta.cls()
            return meta.instance

    def get_all(self) -> list[Plugin]:
        with self._lock:
            result = []
            for name, meta in sorted(self._plugins.items(),
                                     key=lambda x: x[1].priority):
                if meta.enabled:
                    if meta.instance is None:
                        meta.instance = meta.cls()
                    result.append(meta.instance)
            return list(result)

    def list_registered(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "name": name,
                    "cls": meta.cls.__name__,
                    "version": getattr(meta.cls, "version", "1.0.0"),
                    "enabled": meta.enabled,
                    "priority": meta.priority,
                }
                for name, meta in sorted(self._plugins.items(),
                                         key=lambda x: x[1].priority)
            ]

    def enable(self, name: str):
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = True

    def disable(self, name: str):
        with self._lock:
            if name in self._plugins:
                self._plugins[name].enabled = False

    def clear(self):
        with self._lock:
            self._plugins.clear()


_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    return _registry


def plugin(name: str = "", version: str = "1.0.0",
           description: str = "", config: dict = None,
           enabled: bool = True, priority: int = 100):
    def decorator(cls):
        if not issubclass(cls, Plugin):
            raise TypeError(f"{cls.__name__} must be a Plugin subclass")
        if name:
            cls.name = name
        cls.version = version
        cls.description = description
        _registry.register(cls, config=config, enabled=enabled, priority=priority)
        return cls
    return decorator


class HookRunner:
    """Runs all enabled plugin hooks for a given hook point.

    Each method accepts the same args as the plugin hook signature and
    returns the modified values. Error isolation: one failing plugin
    does not break the chain.
    """

    def __init__(self, registry: PluginRegistry = None):
        if registry is not None and not isinstance(registry, PluginRegistry):
            raise TypeError("registry must be a PluginRegistry or None")
        self.registry = registry or _registry

    def run_before_plan(self, goal: str, context: dict) -> tuple[str, dict]:
        for p in self.registry.get_all():
            try:
                goal, context = p.before_plan(goal, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} before_plan failed: {e}")
        return goal, context

    def run_after_plan(self, plan: dict, context: dict) -> tuple[dict, dict]:
        for p in self.registry.get_all():
            try:
                plan, context = p.after_plan(plan, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} after_plan failed: {e}")
        return plan, context

    def run_before_step(self, step: dict, context: dict) -> tuple[dict, dict]:
        for p in self.registry.get_all():
            try:
                step, context = p.before_step(step, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} before_step failed: {e}")
        return step, context

    def run_after_step(self, step: dict, result: dict, context: dict) -> tuple[dict, dict]:
        for p in self.registry.get_all():
            try:
                result, context = p.after_step(step, result, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} after_step failed: {e}")
        return result, context

    def run_before_code_gen(self, step: dict, prompt: str, context: dict) -> tuple[str, dict]:
        for p in self.registry.get_all():
            try:
                prompt, context = p.before_code_gen(step, prompt, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} before_code_gen failed: {e}")
        return prompt, context

    def run_after_code_gen(self, step: dict, code: str, context: dict) -> tuple[str, dict]:
        for p in self.registry.get_all():
            try:
                code, context = p.after_code_gen(step, code, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} after_code_gen failed: {e}")
        return code, context

    def run_on_error(self, step: dict, error: str, context: dict) -> tuple[Optional[str], dict]:
        recovery = None
        for p in self.registry.get_all():
            try:
                r, context = p.on_error(step, error, context)
                if r is not None:
                    recovery = r
            except Exception as e:
                logger.error(f"Plugin {p.name} on_error failed: {e}")
        return recovery, context

    def run_on_summary(self, summary: str, context: dict) -> tuple[str, dict]:
        for p in self.registry.get_all():
            try:
                summary, context = p.on_summary(summary, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} on_summary failed: {e}")
        return summary, context

    def run_on_complete(self, result: dict, context: dict) -> tuple[dict, dict]:
        for p in self.registry.get_all():
            try:
                result, context = p.on_complete(result, context)
            except Exception as e:
                logger.error(f"Plugin {p.name} on_complete failed: {e}")
        return result, context
