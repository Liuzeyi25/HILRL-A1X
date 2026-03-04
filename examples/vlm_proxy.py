import os
import re
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class VLMProxy:
    def __init__(self, model_dir, device="cuda"):
        print("="*50)
        print("正在加载 Qwen3-VL 作为 Reward Proxy...")
        self.device = device
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, 
            device_map=device, 
            torch_dtype="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
        
        self.system_prompt = """You are an expert robot execution state annotator.
        Your task is to analyze a complete robot manipulation video and provide detailed timeline segmentation.
        Output Format strictly follows:
        - [start% - end%]: Step description
        TASK_STATUS: Success/Failure
        """
        print("Qwen3-VL Reward Proxy 加载完毕！")
        print("="*50)

    def get_trajectory_reward(self, main_video_path, task_instruction, traj_len):
        """
        调用 VLM 获取分段，并将百分比映射回轨迹步数，返回每一步的 Dense Reward 增量字典
        """
        user_prompt = f"Task Instruction: {task_instruction}\nAnalyze the complete robot manipulation video and provide frame-by-frame stage analysis."
        
        user_content = [
            {"type": "video", "video": main_video_path, "max_pixels": 128*32*32, "max_frames": 32},
            # {"type": "video", "video": wrist_video_path, "max_pixels": 128*32*32, "max_frames": 32},
            {"type": "text", "text": user_prompt},
        ]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
        )
        
        # 处理视频元数据
        if video_inputs is not None:
            video_inputs_processed, video_metadatas = [], []
            for item in video_inputs:
                if isinstance(item, tuple) and len(item) == 2:
                    video_inputs_processed.append(item[0])
                    video_metadatas.append(item[1])
                else:
                    video_inputs_processed.append(item)
                    video_metadatas.append(None)
            video_inputs = video_inputs_processed
        else:
            video_metadatas = None
        
        inputs = self.processor(
            text=[text],
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.8)
        
        output = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()

        print(f"\n[VLM Output]:\n{output}\n")
        return self._parse_output_to_rewards(output, traj_len)

    def _parse_output_to_rewards(self, vlm_output, traj_len):
        """
        解析输出并将 reward 赋予对应的 step index。
        逻辑：将 1.0 的进度分平均分给各个段，段内通过 MC Discount 方式实现密集化。
        最后成功步 +5。
        """
        step_rewards = {i: 0.0 for i in range(traj_len)}
        
        # 1. 检查是否成功，失败则直接返回全0
        is_success = "TASK_STATUS: Success" in vlm_output
        if not is_success:
            print("  -> 任务失败，全步长 Reward 为 0")
            return step_rewards

        # 2. 提取时间段
        pattern = r"\[(\d+)%\s*-\s*(\d+)%\]:\s*(.*?)(?=\n|$)"
        segments = re.findall(pattern, vlm_output)
        
        if not segments:
            # 如果没匹配到段落但成功了，就把整个轨迹当成一段
            segments = [("0", "100", "task")]

        num_segments = len(segments)
        reward_per_segment = 1.0 / num_segments
        gamma = 0.95 # MC 折扣因子，用于段内密集化分布
        
        last_end_idx = 0
        for start_pct, end_pct, desc in segments:
            # 计算当前段的起始和结束索引
            start_idx = last_end_idx
            end_idx = int(traj_len * float(end_pct) / 100.0)
            end_idx = min(max(start_idx + 1, end_idx), traj_len) # 确保至少有1步
            
            segment_steps = end_idx - start_idx
            
            # 在段内生成 MC 风格的权重: w_i = gamma^(dist_to_end)
            # 这样越靠近段终点，reward 越高
            weights = [gamma**(segment_steps - 1 - i) for i in range(segment_steps)]
            sum_w = sum(weights)
            
            for i in range(segment_steps):
                # 将 reward_per_segment 按权重分配到每一步
                step_val = (weights[i] / sum_w) * reward_per_segment
                step_rewards[start_idx + i] = step_val
            
            print(f"  -> 段分配 [{start_idx}-{end_idx-1}]: {desc} (总分: {reward_per_segment:.3f})")
            last_end_idx = end_idx

        # 3. 处理可能遗漏的尾部 step (如果百分比没到100%)
        if last_end_idx < traj_len:
            remain_steps = traj_len - last_end_idx
            for i in range(last_end_idx, traj_len):
                step_rewards[i] = 0.0

        # 4. 成功奖励：最后一步强行为 +5 (不影响之前的 0~1 分布)
        final_step = traj_len - 1
        step_rewards[final_step] = 5.0
        print(f"  -> 最终成功奖励已注入 step {final_step}: +5.0")

        return step_rewards