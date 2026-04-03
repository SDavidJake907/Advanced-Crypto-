import pytest
import tempfile
from pathlib import Path

@pytest.fixture(autouse=True, scope="session")
def isolate_runtime_overrides():
    """Ensure tests don't read or write to live production configs."""
    import core.config.runtime
    import shutil
    
    # Needs to be inside project ROOT for .relative_to(ROOT) to work
    temp_dir = core.config.runtime.ROOT / "tests" / ".tmp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    original_overrides = core.config.runtime.RUNTIME_OVERRIDES_PATH
    original_proposals = core.config.runtime.RUNTIME_OVERRIDE_PROPOSALS_PATH
    
    core.config.runtime.RUNTIME_OVERRIDES_PATH = temp_dir / "runtime_overrides.json"
    core.config.runtime.RUNTIME_OVERRIDE_PROPOSALS_PATH = temp_dir / "runtime_override_proposals.json"
    
    import os
    env_backups = {}
    for key in core.config.runtime._SETTING_SPECS:
        if key in os.environ:
            env_backups[key] = os.environ.pop(key)
    
    yield
    
    for key, val in env_backups.items():
        os.environ[key] = val
        
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    core.config.runtime.RUNTIME_OVERRIDES_PATH = original_overrides
    core.config.runtime.RUNTIME_OVERRIDE_PROPOSALS_PATH = original_proposals
