# Record Demos 数据记录修复说明

## 🔍 问题分析

对比 `record_demos_octo.py`（旧版）和 `record_demos_octo_manual_new.py`（新版），发现新版在手动成功模式下存在严重的数据记录问题。

---

## ❌ 发现的问题

### 1. **手动成功时 info 被错误覆盖**
**位置**: 第 183-190 行

**问题代码**:
```python
if manual_success_before_step:
    done = True
    next_obs = obs
    rew = 0
    info = {"succeed": False}  # ❌ 错误！丢失所有环境信息
```

**影响**:
- 丢失了 `intervene_action_eef` 等关键信息
- 丢失了环境返回的所有其他状态
- `info["succeed"]` 被设为 False 后需要再次修改，逻辑混乱

**修复后**:
```python
if manual_success_before_step:
    done = True
    truncated = False
    next_obs = obs
    rew = 0
    info = copy.deepcopy(info) if 'info' in locals() else {}
    info["succeed"] = True
    info["manual_success"] = True
```

---

### 2. **truncated 变量缺失**
**位置**: 第 183-190 行

**问题**:
- 手动成功路径中没有定义 `truncated` 变量
- 导致后续代码可能引用未定义的变量

**修复**: 添加 `truncated = False`

---

### 3. **跳过无效帧时未更新 obs**
**位置**: 第 251-260 行

**问题代码**:
```python
if not action_valid:
    # 检查手动成功...
    else:
        obs = next_obs  # ❌ 只在 else 分支更新
        continue
```

**影响**:
- 如果触发手动成功，obs 没有更新，下一帧会使用旧的观测值

**修复后**:
```python
if not action_valid:
    # 🔧 修复：无效数据时也要更新 obs，避免使用旧数据
    obs = next_obs if 'next_obs' in locals() else obs
    
    if FLAGS.manual_success and keyboard_listener and keyboard_listener.manual_success:
        print("\n🎯 检测到手动成功标志（数据无效但强制结束）")
        done = True
        # 跳转到 episode 结束处理
    else:
        continue
```

---

### 4. **重复的 episode 结束处理代码**
**位置**: 第 251-301 行（原代码）

**问题**:
- 在 `if not action_valid` 分支内有完整的 episode 结束处理逻辑
- 与后面正常的 episode 结束处理代码重复
- 代码维护困难，容易出现不一致

**修复**: 
- 删除重复代码
- 统一使用后面的 episode 结束处理逻辑
- 在无效数据分支只设置 `done = True`，让代码流程继续到统一的处理点

---

## ✅ 修复后的数据流程

### 正常流程（action_valid = True）
```
1. env.step() → 获取 next_obs, rew, done, truncated, info
2. 从 info 中提取 intervene_action_eef/intervene_action
3. 创建 transition 字典，包含所有完整信息
4. 添加到 trajectory
5. 更新 obs = next_obs
6. 如果 done，进入 episode 结束处理
```

### 手动成功流程（manual_success_before_step = True）
```
1. 检测到手动成功标志 且 trajectory 有数据
2. 设置 done=True, truncated=False, next_obs=obs, rew=0
3. 保留并更新 info: succeed=True, manual_success=True
4. 跳过 env.step()
5. 提取 actions（使用最后一帧的 actions 或从 info）
6. 如果 action_valid，创建 transition 并添加到 trajectory
7. 更新 obs = next_obs
8. 进入 episode 结束处理（done=True）
```

### 无效数据流程（action_valid = False）
```
1. 更新 obs = next_obs（避免使用旧数据）
2. 如果检测到手动成功标志：
   - 设置 done=True
   - 继续流程，进入 episode 结束处理
3. 否则：
   - continue，跳过此帧，继续采集数据
```

---

## 📋 关键数据字段确认

每个 transition 必须包含的字段：

```python
transition = {
    "observations": obs,              # 当前观测（字典）
    "actions": actions,               # 动作（numpy array, 7维或14维）
    "next_observations": next_obs,    # 下一步观测（字典）
    "rewards": rew,                   # 奖励（float）
    "masks": 1.0 - done,              # mask（1.0 或 0.0）
    "dones": done,                    # done标志（bool）
    "infos": info,                    # 信息字典
}
```

**info 字典应包含**:
- `succeed`: 成功标志（bool）
- `intervene_action_eef`: EEF空间的干预动作（如果有 Gello）
- `intervene_action`: 关节空间的干预动作（备用）
- `manual_success`: 手动成功标志（如果是手动触发）
- `data_valid`: 数据有效性标志
- 其他环境返回的信息

---

## 🔄 对比：旧版 vs 新版（修复后）

### 旧版 (record_demos_octo.py) 
✅ **优点**:
- 逻辑简单清晰
- 数据记录完整
- 立即处理 embeddings

❌ **缺点**:
- 没有手动成功功能
- embeddings 处理慢，阻塞数据采集

### 新版 (record_demos_octo_manual_new.py) - 修复后
✅ **优点**:
- 支持手动成功标记（按 's' 键）
- 延后处理 embeddings，提高采集效率
- 数据验证更完善
- 保存单独的轨迹文件

✅ **修复后新增**:
- 正确保留环境返回的所有 info 信息
- 添加 `manual_success` 标记区分手动和自动成功
- 统一的 episode 结束处理逻辑
- 正确处理无效数据情况下的 obs 更新

---

## 🧪 验证建议

运行修复后的代码时，建议检查：

1. **检查 transition 数据完整性**:
```python
# 在保存前添加验证
for i, trans in enumerate(trajectory):
    assert "observations" in trans
    assert "actions" in trans and trans["actions"] is not None
    assert "next_observations" in trans
    assert "rewards" in trans
    assert "masks" in trans
    assert "dones" in trans
    assert "infos" in trans
    assert "succeed" in trans["infos"]
```

2. **检查 info 内容**:
```python
# 手动成功的轨迹应该有标记
if manual_success:
    assert trajectory[-1]["infos"]["succeed"] == True
    assert trajectory[-1]["infos"].get("manual_success") == True
```

3. **检查动作维度**:
```python
# 单臂7维，双臂14维
assert len(trans["actions"]) in [7, 14]
assert not np.any(np.isnan(trans["actions"]))
```

---

## 📝 修改总结

**文件**: `examples/record_demos_octo_manual_new.py`

**修改位置**:
1. 第 183-190 行: 手动成功时正确保留和更新 info
2. 第 251-267 行: 删除重复的 episode 处理代码，统一流程
3. 添加 truncated 变量定义
4. 修复无效数据时的 obs 更新逻辑

**向后兼容性**: ✅ 完全兼容
- 不影响自动成功检测
- 不影响正常数据采集流程
- 只修复了手动成功模式的 bug

---

**修复日期**: 2026-02-01
**修复者**: GitHub Copilot
