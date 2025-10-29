#!/bin/bash
set -e

# Ensure we're in x86_64 mode
if [ "$(uname -m)" != "x86_64" ]; then
  echo "❌ Not running under Rosetta. Run again with: arch -x86_64 zsh setup_rosetta_env.sh"
  exit 1
fi

# 1. Ensure Rosetta installed
softwareupdate --install-rosetta --agree-to-license || true

# 2. Install Intel Homebrew if missing
if [ ! -d "/usr/local/Homebrew" ]; then
  echo "Installing Intel Homebrew..."
  arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 3. Add Intel brew to path
eval "$(/usr/local/Homebrew/bin/brew shellenv)"

# 4. Dependencies
arch -x86_64 brew install gcc@11 cmake patchelf wget

# 5. Intel Python
arch -x86_64 pyenv install 3.10.16 -s
arch -x86_64 pyenv virtualenv 3.10.16 rob831-x86 || true
export PYENV_VERSION=rob831-x86

# 6. Download correct MuJoCo 2.1.0 archive
mkdir -p ~/.mujoco && cd ~/.mujoco
wget -O mujoco210-macos.tar.gz https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz
tar -xzf mujoco210-macos.tar.gz
mv mujoco210 mujoco-2.1.0 || mv mujoco-2.1.0 mujoco210

export CC=/opt/homebrew/bin/gcc-11

# 7. Symlinks for mujoco-py
ln -sf ~/.mujoco/mujoco210/lib/libmujoco210.dylib ~/.mujoco/mujoco210/bin/libmujoco210.dylib
ln -sf /usr/local/lib/libglfw.3.dylib ~/.mujoco/mujoco210/bin/libglfw.3.dylib || true

# 8. Env vars
echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$DYLD_LIBRARY_PATH' >> ~/.zshrc
echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.zshrc
echo 'export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt' >> ~/.zshrc

# 9. Install mujoco-py under x86 Python
arch -x86_64 pip install --upgrade pip setuptools wheel
arch -x86_64 pip install 'mujoco-py<2.2,>=2.1'

# 10. Test
arch -x86_64 python -c "import mujoco_py; print('✅ mujoco-py works under Rosetta')"
