#!/bin/bash

# 1. Ensure Rosetta is installed
softwareupdate --install-rosetta --agree-to-license

# 2. Install Intel Homebrew if not already
if [ ! -d "/usr/local/Homebrew" ]; then
  arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 3. Add Intel Homebrew to path for this shell
eval "$(/usr/local/Homebrew/bin/brew shellenv)"

# 4. Install Intel dependencies
arch -x86_64 brew install gcc@9 cmake patchelf

# 5. Install Intel Python 3.10 with pyenv
arch -x86_64 pyenv install 3.10.16 -s
arch -x86_64 pyenv virtualenv 3.10.16 rob831-x86
arch -x86_64 pyenv activate rob831-x86

# 6. Download MuJoCo 2.1.0 Intel build
mkdir -p ~/.mujoco
cd ~/.mujoco
curl -L -o mujoco210-macos.tar.gz \
  https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco-2.1.0-macos-x86_64.tar.gz
tar -xzf mujoco210-macos.tar.gz
mv mujoco-2.1.0 mujoco210

# 7. Symlinks for mujoco-py
ln -sf ~/.mujoco/mujoco210/lib/libmujoco.2.1.0.dylib ~/.mujoco/mujoco210/bin/libmujoco210.dylib
ln -sf /usr/local/lib/libglfw.3.dylib ~/.mujoco/mujoco210/bin/libglfw.3.dylib

# 8. Environment variables
echo 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH' >> ~/.zshrc
echo 'export DYLD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$DYLD_LIBRARY_PATH' >> ~/.zshrc
echo 'export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210' >> ~/.zshrc
echo 'export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mjkey.txt' >> ~/.zshrc

# 9. Install mujoco-py
arch -x86_64 pip install 'mujoco-py<2.2,>=2.1'

# 10. Test
arch -x86_64 python -c "import mujoco_py; print('✅ mujoco-py works under Rosetta')"
