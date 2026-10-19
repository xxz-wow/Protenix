# Copyright 2026 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_prebuilt_extension(module_name: str, directory: str | Path) -> ModuleType:
    """Load a colocated prebuilt extension without compiling a fallback."""
    extension_path = Path(directory).resolve() / f"{module_name}.so"
    if not extension_path.is_file():
        raise FileNotFoundError(
            f"Required prebuilt extension {module_name!r} was not found at "
            f"{extension_path}. Runtime compilation is disabled."
        )

    loaded = sys.modules.get(module_name)
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if loaded_path == extension_path:
            return loaded
        raise ImportError(
            f"Extension {module_name!r} is already loaded from {loaded_path}, "
            f"but {extension_path} is required."
        )

    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot create an import specification for prebuilt extension "
            f"{module_name!r} at {extension_path}."
        )

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        if sys.modules.get(module_name) is locals().get("module"):
            sys.modules.pop(module_name, None)
        raise ImportError(
            f"Failed to load prebuilt extension {module_name!r} from "
            f"{extension_path}."
        ) from exc
    return module
