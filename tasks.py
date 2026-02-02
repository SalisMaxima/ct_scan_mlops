"""
Invoke tasks for CT Scan MLOps project.

Tasks are organized into namespaces for better organization:
- core:    Environment setup and maintenance (bootstrap, sync, setup-dev)
- data:    Data management (download, preprocess, extract-features, stats, validate)
- train:   Training and sweeps (train, train-dual, sweep, sweep-agent)
- eval:    Model evaluation (analyze, benchmark, profile, model-info)
- quality: Code quality (ruff, test, ci, security-check)
- deploy:  Deployment (promote-model, export-onnx, api, frontend)
- docker:  Docker operations (build, train, api, clean)
- monitor: Monitoring (extract-stats, check-drift)
- git:     Git operations (status, commit, branch)
- dvc:     DVC operations (pull, push, add)
- docs:    Documentation (build, serve)
- utils:   Utilities (clean-all, env-info, check-gpu, port-check)

Usage:
    invoke <namespace>.<task> [options]

Examples:
    invoke core.setup-dev
    invoke train.train --args "model=resnet18"
    invoke quality.ci
    invoke docker.build
    invoke data.download

For backward compatibility, you can still use the old flat structure
by importing tasks from the namespace directly.
"""

import importlib.util
from pathlib import Path

from invoke import Collection


# Helper to load module from file
def load_module_from_file(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load all task modules from tasks/ directory
tasks_dir = Path(__file__).parent / "tasks"

core = load_module_from_file("core", tasks_dir / "core.py")
data = load_module_from_file("data", tasks_dir / "data.py")
train = load_module_from_file("train", tasks_dir / "train.py")
eval_mod = load_module_from_file("eval", tasks_dir / "eval.py")
quality = load_module_from_file("quality", tasks_dir / "quality.py")
deploy = load_module_from_file("deploy", tasks_dir / "deploy.py")
docker = load_module_from_file("docker", tasks_dir / "docker.py")
monitor = load_module_from_file("monitor", tasks_dir / "monitor.py")
git_tasks = load_module_from_file("git_tasks", tasks_dir / "git_tasks.py")
dvc_tasks = load_module_from_file("dvc_tasks", tasks_dir / "dvc_tasks.py")
docs = load_module_from_file("docs", tasks_dir / "docs.py")
utils = load_module_from_file("utils", tasks_dir / "utils.py")

# Create the root namespace
namespace = Collection()

# Add all sub-namespaces
namespace.add_collection(Collection.from_module(core), name="core")
namespace.add_collection(Collection.from_module(data), name="data")
namespace.add_collection(Collection.from_module(train), name="train")
namespace.add_collection(Collection.from_module(eval_mod), name="eval")
namespace.add_collection(Collection.from_module(quality), name="quality")
namespace.add_collection(Collection.from_module(deploy), name="deploy")
namespace.add_collection(Collection.from_module(docker), name="docker")
namespace.add_collection(Collection.from_module(monitor), name="monitor")
namespace.add_collection(Collection.from_module(git_tasks), name="git")
namespace.add_collection(Collection.from_module(dvc_tasks), name="dvc")
namespace.add_collection(Collection.from_module(docs), name="docs")
namespace.add_collection(Collection.from_module(utils), name="utils")
