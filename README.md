# Package management
## Installation
Please install `uv`, a python package manager [(link)](https://astral.sh/blog/uv), using the following command
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Directory setup
1. Activate the virtual environment
```zsh
source .venv/bin/activate
```
2. Find a directory of `site-packages`. It's something like `/Users/pakhapoomsarapat/Workspaces/simulating-agents/.venv/lib/python3.10/site-packages` for example.
```zsh
python -m site
```
3. Add a path to `custom_path.pth` inside the `site-packages`'s directory.
```zsh
echo path/to/project > path/to/site-packages/custom_path.pth
```
For example,
```zsh
echo "/Users/pakhapoomsarapat/Workspaces/simulating-agents" > /Users/pakhapoomsarapat/Workspaces/simulating-agents/.venv/lib/python3.10/site-packages/custom_path.pth
```

## Exploration
For running a python script, please specify `uv run` followed by the python file, such as
```zsh
uv run path/to/py/file.py
```

# Project structure
Everything is inside `src` for now.