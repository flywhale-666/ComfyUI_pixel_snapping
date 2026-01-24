import cv2
import numpy as np
import torch


class ColorAlignmentNode:
    """
    强力颜色对齐节点
    
    智能检测相同区域，自动对齐颜色
    特性：
    - 智能阈值检测（替代固定百分比）
    - 饱和度加权调整（灰色区域不影响颜色通道）
    - 平滑的边缘融合
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),  # 待调整的图像
                "reference_image": ("IMAGE",),  # 参考图像
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_image",)
    FUNCTION = "align_color"
    CATEGORY = "image/color"
    
    def align_color(self, target_image, reference_image, strength):
        """
        基于对应关系的颜色对齐
        """
        target = target_image[0].cpu().numpy()
        reference = reference_image[0].cpu().numpy()
        
        target_uint8 = (target * 255).astype(np.uint8)
        reference_uint8 = (reference * 255).astype(np.uint8)
        
        aligned, weight_map = self._region_aware_color_transfer(target_uint8, reference_uint8, strength)
        
        # 强度混合
        if strength != 1.0 and weight_map is None:
            if strength > 1.0:
                aligned = cv2.addWeighted(aligned, strength, target_uint8, 1 - strength, 0)
                aligned = np.clip(aligned, 0, 255).astype(np.uint8)
            else:
                aligned = cv2.addWeighted(target_uint8, 1 - strength, aligned, strength, 0)
        
        aligned_float = aligned.astype(np.float32) / 255.0
        result = torch.from_numpy(aligned_float).unsqueeze(0)
        
        print(f"✓ 颜色对齐完成, 强度={strength:.2f}")
        
        return (result,)
    
    def _region_aware_color_transfer(self, target, reference, strength=1.0):
        """
        区域感知颜色对齐
        
        特性：
        - 智能阈值检测相同区域
        - 饱和度加权调整a/b通道
        - 平滑权重图融合
        """
        print("=== Powerful Color Alignment ===")
        
        h, w = target.shape[:2]
        img_size = min(h, w)
        
        # 自适应参数
        kernel_size = max(int(img_size * 0.015), 11)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blur_size1 = max(int(img_size * 0.45), 201)
        if blur_size1 % 2 == 0:
            blur_size1 += 1
        blur_size2 = max(int(img_size * 0.32), 151)
        if blur_size2 % 2 == 0:
            blur_size2 += 1
        blur_size3 = max(int(img_size * 0.22), 101)
        if blur_size3 % 2 == 0:
            blur_size3 += 1
        blur_size4 = max(int(img_size * 0.15), 71)
        if blur_size4 % 2 == 0:
            blur_size4 += 1
        
        print(f"  图像尺寸: {w}×{h}")
        
        # 转到 Lab 空间
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 计算饱和度权重图（用于a/b通道的加权调整）
        saturation_weight = self._compute_saturation_weight(target)
        print(f"  饱和度权重范围: [{saturation_weight.min():.2f}, {saturation_weight.max():.2f}]")
        
        # 1. 检测差异区域
        print("  检测变化区域...")
        diff_map = self._detect_difference(target, reference)
        
        # 2. 使用智能阈值检测
        background_mask, auto_threshold = self._find_background_auto(diff_map, kernel_size)
        
        background_ratio = background_mask.sum() / background_mask.size
        print(f"  相同区域占比: {background_ratio*100:.1f}% (自动阈值)")
        
        # 3. 对相同区域做直方图匹配
        print("  对相同区域进行直方图匹配...")
        
        if background_mask.sum() < 100:
            print("  警告：相同区域太小，使用全局对齐")
            background_mask = np.ones_like(background_mask)
        
        bg_corrected_lab = target_lab.copy()
        
        for i in range(3):
            target_bg_channel = target_lab[:, :, i][background_mask].flatten()
            ref_bg_channel = reference_lab[:, :, i][background_mask].flatten()
            
            if len(target_bg_channel) == 0:
                continue
            
            # 直方图匹配
            target_hist, _ = np.histogram(target_bg_channel, bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(ref_bg_channel, bins=256, range=(0, 256))
            
            target_cdf = np.cumsum(target_hist).astype(np.float64)
            target_cdf = target_cdf / (target_cdf[-1] + 1e-6)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-6)
            
            target_values = np.linspace(0, 255, 256)
            matched_values = np.zeros(256)
            
            for j in range(256):
                target_cdf_val = target_cdf[j] if j < len(target_cdf) else 1.0
                idx = np.searchsorted(ref_cdf, target_cdf_val)
                idx = min(idx, 255)
                matched_values[j] = idx
            
            channel_data = target_lab[:, :, i]
            channel_clipped = np.clip(channel_data, 0, 255)
            bg_corrected_lab[:, :, i] = np.interp(
                channel_clipped.flatten(),
                target_values,
                matched_values
            ).reshape(target_lab.shape[:2])
            
            print(f"    通道{i}: [{target_bg_channel.min():.1f}, {target_bg_channel.max():.1f}] "
                  f"-> [{ref_bg_channel.min():.1f}, {ref_bg_channel.max():.1f}]")
        
        # 4. 创建权重图并应用
        print("  创建权重图...")
        result_lab = target_lab.copy()
        weight_map = background_mask.astype(np.float32)
        
        foreground_mask = ~background_mask
        if foreground_mask.sum() > 0:
            dist_transform = cv2.distanceTransform(foreground_mask.astype(np.uint8), cv2.DIST_L2, 5)
            if dist_transform.max() > 0:
                dist_norm = dist_transform / dist_transform.max()
                decay = np.exp(-dist_norm[foreground_mask] * 0.8)
                weight_map[foreground_mask] = np.clip(0.70 + decay * 0.24, 0.70, 0.94)
        
        # 平滑权重图
        weight_map = cv2.GaussianBlur(weight_map, (blur_size1, blur_size1), 0)
        weight_map = cv2.GaussianBlur(weight_map, (blur_size2, blur_size2), 0)
        weight_map = cv2.GaussianBlur(weight_map, (blur_size3, blur_size3), 0)
        weight_map = cv2.GaussianBlur(weight_map, (blur_size4, blur_size4), 0)
        
        adjusted_weight_map = weight_map * strength
        
        # L通道：正常调整
        result_lab[:, :, 0] = (
            target_lab[:, :, 0] * (1 - adjusted_weight_map) +
            bg_corrected_lab[:, :, 0] * adjusted_weight_map
        )
        
        # a/b通道：按饱和度加权调整
        # 灰色区域（饱和度低）：调整幅度小
        # 有颜色区域（饱和度高）：调整幅度大
        for i in [1, 2]:  # a和b通道
            # 计算该通道的调整量
            channel_diff = bg_corrected_lab[:, :, i] - target_lab[:, :, i]
            if background_mask.any():
                bg_diffs = channel_diff[background_mask]
                if bg_diffs.size > 0:
                    low = np.percentile(bg_diffs, 5.0)
                    high = np.percentile(bg_diffs, 95.0)
                    channel_diff = np.clip(channel_diff, low, high)
            
            # 调整量 × 饱和度权重
            weighted_diff = channel_diff * saturation_weight
            
            # 应用加权后的调整
            result_lab[:, :, i] = (
                target_lab[:, :, i] * (1 - adjusted_weight_map) +
                (target_lab[:, :, i] + weighted_diff) * adjusted_weight_map
            )
        
        result_lab = np.clip(result_lab, 0, 255)
        
        print(f"  L通道: 正常调整")
        print(f"  a/b通道: 按饱和度加权调整")
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        result_denoised = cv2.medianBlur(result, 5)
        diff = cv2.absdiff(result, result_denoised)
        diff_sum = diff[:, :, 0].astype(np.int16) + diff[:, :, 1].astype(np.int16) + diff[:, :, 2].astype(np.int16)
        dark_sum = result[:, :, 0].astype(np.int16) + result[:, :, 1].astype(np.int16) + result[:, :, 2].astype(np.int16)
        pepper_mask = (diff_sum > 18) & (dark_sum < 90)
        if pepper_mask.any():
            result[pepper_mask] = result_denoised[pepper_mask]
        
        print("✓ 区域感知颜色对齐完成")
        
        return result, adjusted_weight_map
    
    def _detect_difference(self, img1, img2):
        """检测两张图的差异"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        img_size = min(img1.shape[:2])
        blur_size = max(int(img_size * 0.02), 11)
        if blur_size % 2 == 0:
            blur_size += 1
        
        mean1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 0)
        mean2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 0)
        
        var1 = cv2.GaussianBlur(gray1**2, (blur_size, blur_size), 0) - mean1**2
        var2 = cv2.GaussianBlur(gray2**2, (blur_size, blur_size), 0) - mean2**2
        
        mean_diff = np.abs(mean1 - mean2)
        var_diff = np.abs(np.sqrt(np.maximum(var1, 0)) - np.sqrt(np.maximum(var2, 0)))
        
        diff = mean_diff + var_diff * 0.5
        
        return diff
    
    def _find_background_auto(self, diff_map, kernel_size):
        """
        智能阈值检测
        
        思路：
        1. 先用种子方法计算初始阈值
        2. 如果相同区域太小（<40%），自动放宽阈值
        """
        # 归一化
        diff_min = diff_map.min()
        diff_max = diff_map.max()
        diff_norm = (diff_map - diff_min) / (diff_max - diff_min + 1e-6)
        
        print(f"  差异范围: [{diff_min:.4f}, {diff_max:.4f}]")
        
        # 方法：找差异最小的像素作为种子，分析它们的分布
        seed_threshold = np.percentile(diff_norm, 10)
        seed_mask = diff_norm <= seed_threshold
        
        if seed_mask.sum() < 100:
            seed_threshold = np.percentile(diff_norm, 20)
            seed_mask = diff_norm <= seed_threshold
        
        # 计算种子区域的统计
        seed_diffs = diff_norm[seed_mask]
        seed_mean = np.mean(seed_diffs)
        seed_std = np.std(seed_diffs)
        
        # 初始阈值 = 种子均值 + 3倍标准差
        threshold = seed_mean + 3 * seed_std
        threshold = max(threshold, seed_threshold * 1.5)
        
        print(f"  种子阈值={seed_threshold:.4f}, 种子均值={seed_mean:.4f}, 种子标准差={seed_std:.4f}")
        print(f"  初始阈值={threshold:.4f}")
        
        # 检查相同区域占比
        background_mask = diff_norm < threshold
        bg_ratio = background_mask.sum() / background_mask.size
        
        # 如果相同区域太小（<40%），逐步放宽阈值
        min_bg_ratio = 0.40
        while bg_ratio < min_bg_ratio and threshold < 0.8:
            threshold *= 1.2  # 放宽20%
            background_mask = diff_norm < threshold
            bg_ratio = background_mask.sum() / background_mask.size
            print(f"  相同区域占比{bg_ratio*100:.1f}%太小，放宽阈值到{threshold:.4f}")
        
        print(f"  最终阈值={threshold:.4f}, 相同区域占比={bg_ratio*100:.1f}%")
        
        # 形态学操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        background_mask = cv2.morphologyEx(background_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.dilate(background_mask, kernel, iterations=1)
        background_mask = background_mask.astype(bool)
        
        return background_mask, threshold
    
    def _compute_saturation_weight(self, img):
        """
        计算饱和度权重图
        
        饱和度高的区域权重接近1，饱和度低的区域（灰色）权重接近0
        用于a/b通道的加权调整
        """
        # 转到HSV空间获取饱和度
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)  # 0-255
        
        # 归一化到0-1
        sat_norm = saturation / 255.0
        
        # 使用sigmoid函数做平滑映射
        # 饱和度<30/255≈0.12 的区域权重很低
        # 饱和度>80/255≈0.31 的区域权重接近1
        # 中心点设在0.2，斜率10
        weight = 1.0 / (1.0 + np.exp(-10 * (sat_norm - 0.2)))
        
        # 稍微平滑一下，避免突变
        weight = cv2.GaussianBlur(weight, (5, 5), 0)
        
        return weight

# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "PowerfulColorAlignment": ColorAlignmentNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerfulColorAlignment": "Powerful Color Alignment"
}
