import functools
from typing import Any, Dict, Optional

import torch


class MaybeCompile:
    """A decorator class for conditional torch.compile functionality.

    By default, all decorated functions are compiled.
    After calling clear_compile_targets(), only functions explicitly added
    with set_compile_target() will be compiled.

    Usage:
        maybe_compile = MaybeCompile()

        @maybe_compile
        def func(x):
            return x

        @maybe_compile(fullgraph=True, dynamic=True)
        def func2(x):
            return x

        class MyClass:
            @maybe_compile(fullgraph=True, dynamic=True)
            def method(self, x):
                return x
        is equal to
        maybe_compile.set_compile_target('module.MyClass.method', fullgraph=True, dynamic=True)
    """

    def __init__(self):
        # Mode: 'all' (compile everything) or 'selective' (compile only targets)
        self._mode = "all"

        # Whitelist when in selective mode: {'module_name': {'func_name': compile_kwargs}}
        self._compile_targets = {}

        # Blacklist for functions to never compile: {(module_name, func_name)}
        self._excluded_targets = set()

        # Cache for compiled functions: {func_id: compiled_function}
        self._compiled_funcs = {}

    def __call__(self, fn=None, **compile_kwargs):
        """Apply the decorator with optional torch.compile arguments."""

        def decorator(func):
            original_func = func
            module_name = original_func.__module__
            func_name = original_func.__qualname__
            func_id = f"{module_name}.{func_name}"

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Check if this function should be compiled
                should_compile = self._should_compile(module_name, func_name)

                # Compile if needed and not already compiled
                if should_compile and func_id not in self._compiled_funcs:
                    # Get any specific compile kwargs for this function
                    func_compile_kwargs = compile_kwargs
                    if self._mode == "selective":
                        target_kwargs = self._get_compile_kwargs(module_name, func_name)
                        if target_kwargs:
                            func_compile_kwargs = {**compile_kwargs, **target_kwargs}

                    # Compile the function
                    self._compiled_funcs[func_id] = torch.compile(original_func, **func_compile_kwargs)

                # torch.distributed.breakpoint()
                # print(func_id)
                # Use compiled or original function
                if should_compile and func_id in self._compiled_funcs:
                    return self._compiled_funcs[func_id](*args, **kwargs)
                else:
                    return original_func(*args, **kwargs)

            return wrapper

        # Handle both @maybe_compile and @maybe_compile() usage
        if fn is None:
            return decorator
        return decorator(fn)

    def _should_compile(self, module_name: str, func_qualname: str) -> bool:
        """Check if a function or method should be compiled."""
        # First check if function is explicitly excluded
        target_key = (module_name, func_qualname)
        if target_key in self._excluded_targets:
            return False

        # Then check mode-specific behavior
        if self._mode == "all":
            return True  # Compile everything (except excluded)
        else:  # selective mode
            module_targets = self._compile_targets.get(module_name, {})

            # Handle regular functions
            if func_qualname in module_targets:
                return True

            # Handle class methods
            if "." in func_qualname:
                class_name, method_name = func_qualname.rsplit(".", 1)
                # Check for exact class.method match
                if f"{class_name}.{method_name}" in module_targets:
                    return True
                # Check for class-wide methods
                if class_name in module_targets and method_name in module_targets[class_name]:
                    return True

            return False

    def _get_compile_kwargs(self, module_name: str, func_name: str) -> Optional[Dict[str, Any]]:
        """Get compilation kwargs for a specific function or method."""
        module_targets = self._compile_targets.get(module_name, {})
        return module_targets.get(func_name)

    def set_compile_target(self, target: str, **compile_kwargs):
        """Specify a function to be compiled in selective mode.

        Also removes it from the exclusion list if present.
        """
        # Parse the target string
        parts = target.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid target format: {target}. Expected format: 'module.function' or 'module.class.method'"
            )

        # Handle various formats
        if len(parts) == 2:
            module_name, func_name = parts[0], parts[1]
        else:
            module_name = ".".join(parts[:-2])
            class_and_method = ".".join(parts[-2:])
            func_name = class_and_method

        # Remove from exclusion list if present
        target_key = (module_name, func_name)
        if target_key in self._excluded_targets:
            self._excluded_targets.remove(target_key)

        # Add to compile targets
        if module_name not in self._compile_targets:
            self._compile_targets[module_name] = {}

        self._compile_targets[module_name][func_name] = compile_kwargs or None

        # Clear cached compiled function for this target
        func_id = f"{module_name}.{func_name}"
        if func_id in self._compiled_funcs:
            del self._compiled_funcs[func_id]

    def remove_compile_target(self, target: str):
        """Remove a specific function/method from compilation in any mode.

        In 'all' mode, this adds the function to the exclusion list.
        In 'selective' mode, this removes the function from the targets.

        Args:
            target: The full path of the function to remove from compilation
        """
        # Parse the target string
        parts = target.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid target format: {target}. Expected format: 'module.function' or 'module.class.method'"
            )

        # Handle various formats
        if len(parts) == 2:
            module_name, func_name = parts[0], parts[1]
        else:
            module_name = ".".join(parts[:-2])
            class_and_method = ".".join(parts[-2:])
            func_name = class_and_method

        # Add to exclusion list
        target_key = (module_name, func_name)
        self._excluded_targets.add(target_key)

        # If in selective mode, also remove from compile targets
        if self._mode == "selective":
            if module_name in self._compile_targets and func_name in self._compile_targets[module_name]:
                del self._compile_targets[module_name][func_name]
                # Clean up empty dictionaries
                if not self._compile_targets[module_name]:
                    del self._compile_targets[module_name]

        # Remove from compiled functions cache
        func_id = f"{module_name}.{func_name}"
        if func_id in self._compiled_funcs:
            del self._compiled_funcs[func_id]

    def clear_compile_targets(self):
        """Clear all compile targets and switch to selective mode."""
        self._mode = "selective"
        self._compile_targets = {}
        self._compiled_funcs = {}
        # Don't clear exclusion list - keep it consistent

    def compile_all(self):
        """Switch back to 'all' mode where all decorated functions are compiled
        (except those explicitly excluded)."""
        self._mode = "all"
        self._compile_targets = {}
        self._compiled_funcs = {}
        # Don't clear exclusion list - keep it consistent


# Create a singleton instance
maybe_compile = MaybeCompile()
