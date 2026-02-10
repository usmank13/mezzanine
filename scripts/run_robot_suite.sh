#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for three image-based LeRobot settings.
#
# Usage:
#   ./scripts/run_robot_suite.sh OUT_ROOT
#
# Requires:
#   pip install -e ".[robotics]"  # datasets/hub
#   pip install -e ".[yaml]"      # if using the provided YAML configs

OUT_ROOT=${1:-out_robot_suite}

mezzanine run lerobot_latent_dynamics --out "${OUT_ROOT}/pusht_image" --config configs/lerobot_pusht_image.yml
mezzanine run lerobot_latent_dynamics --out "${OUT_ROOT}/aloha_sim_transfer_cube_scripted_image" --config configs/lerobot_aloha_sim_transfer_cube_scripted_image.yml
mezzanine run lerobot_latent_dynamics --out "${OUT_ROOT}/libero_10_image_subtask" --config configs/lerobot_libero_10_image_subtask.yml

echo "Done. Results written under: ${OUT_ROOT}"
