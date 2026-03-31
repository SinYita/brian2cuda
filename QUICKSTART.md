# Phase 4 快速开始指南

## 📋 概览

本目录包含Issue #269完整实现（Phase 1-4）的测试和基准脚本。

**🎯 目标**: 在异构延迟模式下实现Synaptic Effect Application的多块并行化

**📊 预期改进**: Effect应用时间减少 ≥ 30%

## 📁 关键文件

| 文件 | 大小 | 说明 |
|------|------|------|
| **bench.sh** | 24KB | 完整自动化测试脚本（生产级） |
| **BENCH_README.md** | 11KB | 详细使用和故障排除指南 |
| **todo.md** | 520+ lines | 完整实现记录和阶段进度 |

## 🚀 5分钟快速开始

### 步骤 1: 准备文件
```bash
# 在有GPU的机器上
cd /path/to/brian2cuda

# 确保文件存在
ls -la bench.sh BENCH_README.md
```

### 步骤 2: 运行完整测试
```bash
# 一行命令执行完整测试（推荐）
./bench.sh
```

### 步骤 3: 查看结果
```bash
# 打开最终报告
cat benchmark_results_*/PHASE4_REPORT.md

# 或查看详细日志
tail -100 benchmark_results_*/phase4_test.log
```

## ⚡ 快速命令

| 命令 | 用途 | 时长 |
|------|------|------|
| `./bench.sh` | 完整测试 | 10-30分钟 |
| `./bench.sh --setup-only` | 仅设置环境 | 5-10分钟 |
| `./bench.sh --no-compile` | 跳过编译 | 5-15分钟 |
| `./bench.sh --verbose` | 详细输出 | 10-30分钟 |
| `./bench.sh --help` | 显示帮助 | 即时 |

## 📋 要求检查表

在运行脚本前，确保满足：

- ✅ **GPU可用**: `nvidia-smi` 命令可用
- ✅ **CUDA**: `nvcc --version` 可获得版本信息
- ✅ **Python**: 3.8+ 版本
- ✅ **Internet**: 用于pip安装依赖
- ✅ **磁盘空间**: ≥ 5GB 空闲空间

## 📊 预期结果

### 环境信息输出
```
✓ Found python: /usr/bin/python3
✓ Found nvcc: /usr/local/cuda/bin/nvcc
CUDA version: 11.4
Python version: 3.9.x
```

### 代码验证输出
```
✓ Found pointer m_size declaration
✓ Found m_size_owned flag
✓ Found set_size_address method
✓ Found current_offset calculation
✓ Found bid remapping (partition calculation)
✓ All Phases 1-3 changes verified successfully
```

### 性能对比示例
```
effect              : 250.45 -> 175.32 ms (-30.0%) ✅
propagation         : 100.23 -> 100.45 ms (+0.2%) ✅
neurons             :  50.32 ->  50.12 ms (-0.4%) ✅
total               : 500.00 -> 400.00 ms (-20.0%)
```

## 🔍 结果解读

### Success（成功）
✅ Effect时间 ≥ 30% 减少  
✅ Propagation 时间 ±5% 内  
✅ 所有测试通过  
→ **准备提交PR**

### Warning（警告）
⚠️ Effect时间 20-29% 减少  
⚠️ Propagation 时间 ±5-10%  
→ **需要进一步优化或调查**

### Failure（失败）
❌ Effect时间 < 20% 改进  
❌ Propagation 性能显著下降  
❌ 测试失败  
→ **需要调试，参考BENCH_README.md故障排除部分**

## 📂 输出文件位置

所有结果保存在按时间戳命名的目录中：

```
benchmark_results_20260331_010203/
├── phase4_test.log              # 完整的测试日志
├── PHASE4_REPORT.md             # 最终报告（重要！）
├── run_baseline.py              # Baseline基准脚本
├── run_optimized.py             # 优化基准脚本
├── results_baseline/
│   ├── code_baseline/           # 生成的C++/CUDA代码
│   └── results_baseline/
│       ├── *_profiling.txt      # 性能分析数据
│       └── *_timing.txt         # 执行时间
└── results_optimized/
    ├── code_optimized/
    └── results_optimized/
        ├── *_profiling.txt
        └── *_timing.txt
```

**最重要的文件**: 
- `PHASE4_REPORT.md` - 成功/失败的总结
- `phase4_test.log` - 调试问题时查看

## 🐛 常见问题

### Q: 脚本失败，提示"CUDA compiler not found"
**A**: 安装CUDA Toolkit或确保CUDA在PATH中
```bash
nvcc --version  # 检查
export PATH=/usr/local/cuda/bin:$PATH  # 如果需要
```

### Q: 想要跳过某些步骤怎么办？
**A**: 使用模式选项
```bash
./bench.sh --setup-only       # 仅设置，不测试
./bench.sh --no-compile       # 重用已有build
```

### Q: 测试运行太慢或超时
**A**: 检查GPU和系统状态
```bash
nvidia-smi                    # 检查显存
top                          # 检查CPU使用
free -h                      # 检查内存
```

### Q: 想要自定义网络大小或持续时间
**A**: 编辑bench.sh中的参数
```bash
BRUNEL_N=8000              # 改变神经元数
BRUNEL_DURATION="0.2*second"  # 改变仿真时长
```

## 📝 详细文档

完整的使用指南请参考: [BENCH_README.md](BENCH_README.md)

- 环境设置详解
- 高级用法
- 性能优化建议
- 完整的故障排除指南
- CI/CD集成示例

## 🔄 完整工作流

```
1. 文件准备 → 2. 环境检查 → 3. 代码验证 → 4. 编译
   ↓             ↓            ↓             ↓
5. 回归测试 → 6. 性能基准 → 7. 结果分析 → 8. 报告生成
```

## 📞 支持

**遇到问题？**

1. 查看详细日志: `cat benchmark_results_*/phase4_test.log`
2. 参考故障排除: 见 BENCH_README.md 中的 "Troubleshooting" 部分
3. 使用详细模式: `./bench.sh --verbose`

## ✅ 最终检查表

运行前:
- [ ] GPU可用
- [ ] CUDA安装
- [ ] Python 3.8+
- [ ] 5GB磁盘空间
- [ ] 网络连接

运行后:
- [ ] 检查 PHASE4_REPORT.md
- [ ] 验证 Effect 时间改进 ≥ 30%
- [ ] 确认所有测试通过
- [ ] 查看 phase4_test.log 中是否有警告

---

**准备好了？** 执行以下命令开始:

```bash
chmod +x bench.sh
./bench.sh
```

**预计时长**: 10-30分钟  
**GPU需求**: 任何现代NVIDIA GPU (V100/A100/RTX 等)

---

*文件创建于: 2026-03-31*  
*Issue #269: Multi-block Heterogeneous Delays*
