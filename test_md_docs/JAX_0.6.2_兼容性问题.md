# JAX 0.6.2 兼容性问题检查报告

## 1. `jax.tree_map` 已被移除 ❌

**错误信息**: `AttributeError: jax.tree_map was removed in JAX v0.6.0`

**影响范围**: 全项目约 20+ 处使用

**解决方案**: 
- 替换为 `jax.tree_util.tree_map` (所有版本兼容)
- 或 `jax.tree.map` (JAX 0.4.25+)

### 受影响的文件:

1. **octo/octo/model/octo_model.py** (7处)
   - Line 348: `jax.tree_map(jnp.shape, example_batch["observation"])`
   - Line 353: `jax.tree_map(jnp.shape, example_batch["task"])`
   - Line 362: `jax.tree_map(np.array, dataset_statistics, ...)`
   - Line 463: `jax.tree_map(lambda x: x.tolist(), ...)`
   - Line 492: `jax.tree_map(lambda x: x[:1], example_batch)`
   - Line 537: `jax.tree_map(lambda arr: ("batch", *arr.shape[1:]), ...)`

2. **examples/train_conrft_octo.py** (1处)
   - Line 545: `jax.device_put(jax.tree_map(jnp.array, agent), ...)`

3. **serl_launcher/serl_launcher/common/common.py** (8处)
   - Line 25: `jax.tree_map(lambda x: jax.device_put(x, ...), batch)`
   - Line 121, 134, 168, 210, 225, 226: 多处使用

4. **serl_launcher/serl_launcher/wrappers/chunking.py**
   - Line 12: `return jax.tree_map(...)`

5. **serl_launcher/serl_launcher/wrappers/remap.py**
   - Line 34: `return jax.tree_map(lambda x: observation[x], ...)`

6. **serl_launcher/serl_launcher/data/dataset.py**
   - Line 121: `jax.tree_map(lambda d: jnp.take(d, indx, axis=0), src)`

7. **serl_launcher/serl_launcher/vision/data_augmentations.py** (5处)
   - Lines 178, 186, 265, 284: 多处图像处理中使用

---

## 2. `jax.random.KeyArray` 类型定义问题 ⚠️

**已修复**: octo/octo/utils/typing.py
```python
# 已添加兼容性处理
try:
    PRNGKey = jax.random.KeyArray
except AttributeError:
    PRNGKey = jax.Array
```

---

## 3. `@jax.jit` 装饰器的 `static_argnames` 严格检查 ⚠️

**已修复**: octo/octo/model/octo_model.py
- `sample_transformer` 函数已添加 `sample_shape` 和 `argmax` 参数以匹配装饰器

---

## 4. 其他潜在问题

### 4.1 `jax.tree_multimap` (如果存在)
- JAX 0.4.1+ 已被移除，应使用 `jax.tree_map`

### 4.2 `jax.DeviceArray`
- JAX 0.4.1+ 已移除，应使用 `jax.Array`

### 4.3 `jax.device_put_replicated`
- 在某些版本中可能被弃用，建议检查使用情况

---

## 快速修复方案

### 方案 A: 全局替换 (推荐)

使用以下命令批量替换:

```bash
cd /home/dungeon_master/conrft

# 替换 jax.tree_map -> jax.tree_util.tree_map
find . -name "*.py" -type f -exec sed -i 's/jax\.tree_map/jax.tree_util.tree_map/g' {} +
```

### 方案 B: 创建兼容性包装 (临时方案)

在项目根目录创建 `jax_compat.py`:

```python
import jax
import sys

# JAX 版本兼容性处理
if not hasattr(jax, 'tree_map'):
    # JAX >= 0.6.0
    jax.tree_map = jax.tree_util.tree_map
```

然后在每个文件开头添加:
```python
import jax_compat  # 必须在 import jax 之后
```

### 方案 C: 降级 JAX (不推荐)

```bash
pip install "jax<0.6.0" "jaxlib<0.6.0"
```

---

## 修复优先级

1. **高优先级** (阻塞运行):
   - [ ] 替换所有 `jax.tree_map` → `jax.tree_util.tree_map`

2. **中优先级** (可能影响功能):
   - [x] 修复 `jax.random.KeyArray` (已完成)
   - [x] 修复 `static_argnames` 问题 (已完成)

3. **低优先级** (预防性):
   - [ ] 检查 `DeviceArray` 使用
   - [ ] 检查 `device_put_replicated` 使用

---

## 测试建议

修复后运行以下测试:

```bash
# 1. 测试导入
python -c "from octo.model.octo_model import OctoModel; print('✓ Import OK')"

# 2. 测试加载模型
cd /home/dungeon_master/conrft/examples/experiments/a1x_pick_banana
xvfb-run -a bash run_learner_conrft_pretrain.sh
```
