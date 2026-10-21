# AirSim 本地修改

`airsim-local.patch` 保存本项目在 AirSim 子模块中的 C++ 修改和自建地图、材质、网格；包含动力参数、原生鼠标观察器以及 Blocks 模块依赖。补丁基于子模块提交 `d109f0dc25d4ae425eca7b279da33df5f5fcb42e`，子模块仍使用 Microsoft 官方仓库。

首次克隆后，在项目根目录执行：

```powershell
git submodule update --init --recursive
git -C external/AirSim apply --check ../../patches/airsim-local.patch
git -C external/AirSim apply ../../patches/airsim-local.patch
```

只应用一次。如果当前机器已经有这些修改，可用以下只读检查确认：

```powershell
git -C external/AirSim apply --reverse --check ../../patches/airsim-local.patch
```

应用后，按项目原有流程构建 AirSim 和 Blocks。原生观察器与动力模块的增量构建入口是 `scripts/build_dual_target_observer.ps1` 和 `scripts/build_dual_target_dynamics.ps1`；其中 Unreal 安装路径、编译器版本需要与本机环境一致。GitHub 中不含 Unreal 安装、编译后的 DLL、模型权重或 Python 虚拟环境。

本次代码归档不含 `.env`、日志、`reports/`、`runs/`、运行缓存、每次实验的 `configs/**/source_bundle/` 冻结副本及图表的大型 `input_campaign.json` 原始输入；这些文件保留在本地。历史文档中引用本地实验输出的链接需要原始实验归档才能打开。
