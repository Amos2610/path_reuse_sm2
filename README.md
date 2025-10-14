# path_reuse_sm2
# Path Reuse-based State Machine (PRSM)

**ROS 2 Humble (Ubuntu 22.04)**  
A reusable state-machine framework for industrial robot tasks,  
based on the *Path Reuse Method* proposed by Fumoto *et al.* (2025).  

> 🧩 **Original Paper (Springer Journal of Advanced Mechanical Design, Systems, and Manufacturing)**  
> [Autonomous path-seed selection method based on path reuse for industrial robots](https://link.springer.com/article/10.1007/s10015-025-01054-w)

---

## 🧠 Overview

This repository implements **PRSM (Path Reuse-based State Machine)** —  
a unified skill execution framework that integrates the Path Reuse Method  
with the YASMIN behavior library and web-based visualization via YasminViewer.

It provides a **modular and loop-capable state machine** for industrial robot workflows,  
where each *Skill* (e.g., Grasp, UpdatePathSeed, Put) is dynamically discovered,  
registered as a `State`, and visualized in real-time through a local web interface.

---

### ✨ Key Features

- ✅ Dynamic skill discovery via plugin registration  
- ✅ Seamless integration with [`yasmin`](https://github.com/uleroboticsgroup/yasmin)  
- ✅ Real-time visualization through YasminViewer (`http://localhost:5000`)  
- ✅ Parameterized flow definition using ROS 2 launch parameters  
- ✅ Loop execution support for continuous operation  
- ✅ Minimal coupling: skills can be implemented as standalone ROS 2 nodes  

---

## 📦 Dependencies

This package depends on the following repositories:

- 🔗 **[path_reuse_method](https://github.com/Amos2610/path_reuse_method)** — Core path reuse algorithms  
- 🧩 **yasmin**, **yasmin_ros**, **yasmin_viewer** — State machine and visualization tools  

Required environment:

| Component | Version |
|------------|----------|
| **Ubuntu** | 22.04 LTS |
| **ROS 2** | Humble |
| **Python** | 3.10 |
| **Yasmin** | ≥ 0.6.0 |

---

## 🏗️ Installation

```bash
# Setup workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone dependencies
git clone https://github.com/Amos2610/path_reuse_method.git
git clone https://github.com/Amos2610/path_reuse_sm2.git

# Install ROS dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select path_reuse_sm2 --symlink-install
source install/setup.bash
```