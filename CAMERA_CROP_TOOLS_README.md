# 相机图像裁剪可视化工具

两个用于调试和调整 `IMAGE_CROP` 参数的实用工具。

## 工具概览

### 1. 预设方案查看器 (`visualize_camera_crop.py`)
- 快速浏览预定义的9种裁剪方案
- 实时显示原始图像和裁剪后的对比
- 适合快速筛选合适的裁剪策略

### 2. 交互式调整工具 (`visualize_camera_crop_interactive.py`) ⭐ 推荐
- 使用滑块实时调整裁剪参数
- 左右对比显示原始图像和裁剪效果
- 自动生成 `IMAGE_CROP` 配置代码
- 支持多相机切换
- 适合精确调整裁剪区域

---

## 使用方法

### 方法 1: 预设方案查看器

```bash
# 激活环境
conda activate /home/dungeon_master/conrft/.conda

# 运行脚本
python visualize_camera_crop.py
```

**快捷键:**
- `0-9`: 切换不同的预设裁剪方案
- `s`: 保存当前方案参数到文件
- `q`: 退出

**预设方案:**
- `0`: 原始图像 (无裁剪)
- `1`: 中心裁剪 640x640
- `2`: 上半部裁剪
- `3`: 下半部裁剪
- `4`: 左半部裁剪
- `5`: 右半部裁剪
- `6`: 自定义裁剪 1
- `7`: 自定义裁剪 2 (侧面分类器)
- `8`: 中心 800x600

---

### 方法 2: 交互式调整工具 ⭐

```bash
# 激活环境
conda activate /home/dungeon_master/conrft/.conda

# 运行脚本
python visualize_camera_crop_interactive.py
```

**使用步骤:**

1. **启动后会看到两个窗口并排显示:**
   - 左侧: 原始图像 + 绿色裁剪框
   - 右侧: 裁剪后的结果

2. **使用滑块调整裁剪区域:**
   - `Top`: 从顶部裁剪的像素数
   - `Bottom`: 从底部裁剪的像素数
   - `Left`: 从左侧裁剪的像素数
   - `Right`: 从右侧裁剪的像素数

3. **快捷键:**
   - `c`: 切换到下一个相机
   - `r`: 重置当前相机的裁剪参数
   - `g`: 在终端打印生成的代码
   - `s`: 保存代码到 `image_crop_config.py`
   - `q`: 退出

4. **调整完成后:**
   - 按 `g` 查看生成的代码
   - 按 `s` 保存到文件
   - 复制生成的代码到你的 `config.py` 中

---

## 示例输出

运行交互式工具并调整参数后，会生成如下代码：

```python
IMAGE_CROP = {
    "wrist_1": lambda img: img,
    "side_policy_256": lambda img: img[250:-150, 400:-500],
}
```

直接复制到你的任务配置文件 `examples/experiments/pour_water/config.py` 中即可。

---

## 图像处理流程说明

在机器人环境中，图像处理流程如下：

```
相机读取 (1280x720)
    ↓
裁剪 (IMAGE_CROP)
    ↓
调整大小 (256x256 或 128x128)
    ↓
颜色转换 (BGR → RGB)
    ↓
输入到策略网络
```

---

## 调试技巧

### 1. 找到感兴趣区域
- 使用预设方案查看器快速浏览不同区域
- 识别包含重要信息的图像部分

### 2. 精确调整
- 使用交互式工具微调边界
- 确保裁剪后包含所有关键视觉信息

### 3. 验证不同分辨率
裁剪后的图像会被缩放到：
- 名称包含 `"256"` → 256×256
- 其他 → 128×128

确保裁剪区域的长宽比合理，避免过度拉伸。

### 4. 常见裁剪策略
- **策略相机**: 裁剪掉无关背景，保留机械臂和目标物体
- **分类器相机**: 聚焦在任务目标区域（如碗、盘子等）
- **腕部相机**: 通常保持完整视野

---

## 配置文件修改

修改 `examples/experiments/pour_water/config.py`:

```python
class EnvConfig(DefaultA1XEnvConfig):
    # ... 其他配置 ...
    
    # 相机配置
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "044322073334",
            "dim": (1280, 720),      # 原始分辨率
            "exposure": 10500,
        },
        "side_policy_256": {
            "serial_number": "243222075799",
            "dim": (1280, 720),      # 原始分辨率
            "exposure": 10500,
        },
    }
    
    # 裁剪配置 (使用工具生成的代码)
    IMAGE_CROP = {
        "wrist_1": lambda img: img,
        "side_policy_256": lambda img: img[250:-150, 400:-500],
    }
```

---

## 故障排除

### 相机初始化失败
- 检查相机序列号是否正确
- 确保相机已连接并被系统识别
- 尝试 `rs-enumerate-devices` 查看已连接相机

### 图像显示异常
- 确保裁剪参数不会导致空图像
- 检查 top+bottom < 图像高度
- 检查 left+right < 图像宽度

### 性能问题
- 降低相机分辨率 (如改为 640x480)
- 减少曝光时间
- 关闭其他相机应用

---

## 高级自定义

如需修改相机配置，编辑脚本中的 `camera_config` 字典：

```python
camera_config = {
    "your_camera_name": {
        "serial_number": "your_serial_number",
        "dim": (1280, 720),
        "exposure": 10500,
    },
}
```

---

## 相关文件

- 环境基类: `serl_robot_infra/franka_env/envs/a1x_env.py`
- 相机捕获: `serl_robot_infra/franka_env/camera/rs_capture.py`
- 任务配置: `examples/experiments/pour_water/config.py`

---

**提示**: 建议先用 `visualize_camera_crop_interactive.py` 交互式调整，获得精确的裁剪参数！
