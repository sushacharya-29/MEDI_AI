from pathlib import Path
# Create __init__.py in all module directories
INIT_FILES = [
    'core/__init__.py',
    'ai_engine/__init__.py',
    'services/__init__.py',
    'models/__init__.py',
    'utils/__init__.py',
    'scripts/__init__.py'
]

def create_init_files():
    """Create __init__.py files"""
    for init_file in INIT_FILES:
        Path(init_file).parent.mkdir(parents=True, exist_ok=True)
        Path(init_file).touch()
        print(f"Created {init_file}")