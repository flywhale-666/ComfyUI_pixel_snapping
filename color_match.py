import cv2
import numpy as np
import torch


class ColorMatchNode:
    """
    颜色匹配节点 - 基于对应关系的颜色对齐
    
    核心思路：找到两张图中相同的区域（背景），让这些区域颜色一致
    对于不同的区域（新脸），根据周围的颜色变换进行调整
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_image": ("IMAGE",),  # 待调整的图像（换脸后）
                "reference_image": ("IMAGE",),  # 参考图像（原图）
                "method": ([
                    "region_aware",  # 区域感知对齐（推荐）
                ],),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.5,  # 允许超过1.0，更强力调整
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("matched_image",)
    FUNCTION = "match_color"
    CATEGORY = "image/color"
    
    def match_color(self, target_image, reference_image, method, strength):
        """
        基于对应关系的颜色匹配
        
        Args:
            target_image: 待调整图像（换脸后）[B, H, W, C]
            reference_image: 参考图像（原图）[B, H, W, C]
            method: 匹配方法
            strength: 混合强度 (0-1)
        
        Returns:
            匹配后的图像
        """
        # 只处理 batch 中的第一张
        target = target_image[0].cpu().numpy()  # [H, W, C]
        reference = reference_image[0].cpu().numpy()  # [H, W, C]
        
        # 转换到 0-255 范围
        target_uint8 = (target * 255).astype(np.uint8)
        reference_uint8 = (reference * 255).astype(np.uint8)
        
        # 根据方法选择算法
        if method == "region_aware":
            matched = self._region_aware_color_transfer(target_uint8, reference_uint8)
        else:
            matched = target_uint8
        
        # 强度混合（支持>1.0，让调整更强）
        if strength != 1.0:
            if strength > 1.0:
                # 超过1.0时，进一步强化效果
                # strength=1.5 → 让匹配结果占比150%，原图占比-50%（会被clip限制）
                matched = cv2.addWeighted(matched, strength, target_uint8, 1 - strength, 0)
                matched = np.clip(matched, 0, 255).astype(np.uint8)
            else:
                # 小于1.0时，正常混合
                matched = cv2.addWeighted(target_uint8, 1 - strength, matched, strength, 0)
        
        # 转换回 0-1 范围
        matched_float = matched.astype(np.float32) / 255.0
        
        # 转换回 torch tensor
        result = torch.from_numpy(matched_float).unsqueeze(0)
        
        print(f"✓ 颜色对齐完成: {method}, 强度={strength:.2f}")
        
        return (result,)
    
    def _region_aware_color_transfer(self, target, reference):
        """
        区域感知颜色对齐
        
        核心思路：
        1. 自动检测两张图的差异区域（变化区域 = 新脸，不变区域 = 背景）
        2. 从背景区域学习颜色变换
        3. 背景：强力应用变换
        4. 前景（新脸）：温和应用变换
        """
        print("开始区域感知颜色对齐...")
        
        # 获取图像尺寸，计算自适应参数
        h, w = target.shape[:2]
        img_size = min(h, w)  # 使用短边作为参考
        
        # 自适应参数（相对于图像尺寸）
        kernel_size = max(int(img_size * 0.015), 11)  # ~1.5%的图像大小
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保是奇数
        
        # 极大范围平滑参数（进一步增大）
        blur_size1 = max(int(img_size * 0.45), 201)  # ~45%的图像大小（增大）
        if blur_size1 % 2 == 0:
            blur_size1 += 1
            
        blur_size2 = max(int(img_size * 0.32), 151)  # ~32%的图像大小（增大）
        if blur_size2 % 2 == 0:
            blur_size2 += 1
            
        blur_size3 = max(int(img_size * 0.22), 101)  # ~22%的图像大小（增大）
        if blur_size3 % 2 == 0:
            blur_size3 += 1
            
        blur_size4 = max(int(img_size * 0.15), 71)  # ~15%的图像大小（增大）
        if blur_size4 % 2 == 0:
            blur_size4 += 1
        
        print(f"  图像尺寸: {w}×{h}, 自适应参数: kernel={kernel_size}, blur={blur_size1}/{blur_size2}")
        
        # 转到 Lab 空间
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 1. 检测差异区域
        print("  检测变化区域...")
        diff_map = self._detect_difference(target, reference)
        
        # 归一化到 0-1
        diff_map_norm = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-6)
        
        # 定义背景区域（差异小于阈值）
        # 使用自适应阈值
        threshold = np.percentile(diff_map_norm, 60)  # 60%的像素认为是背景
        background_mask = diff_map_norm < threshold
        
        # 形态学操作，去除噪点（使用自适应kernel）
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        background_mask = cv2.morphologyEx(background_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.dilate(background_mask, kernel, iterations=1)  # 扩张背景
        background_mask = background_mask.astype(bool)
        
        background_ratio = background_mask.sum() / background_mask.size
        print(f"  背景区域占比: {background_ratio*100:.1f}%")
        
        # 2. 对背景区域做强力颜色对齐
        print("  对背景区域进行直方图匹配...")
        
        if background_mask.sum() < 100:
            print("  警告：背景区域太小，使用全局对齐")
            background_mask = np.ones_like(background_mask)
        
        # 对每个通道做直方图匹配（背景区域）
        bg_corrected_lab = target_lab.copy()
        
        for i in range(3):  # L, a, b 三个通道
            # 提取背景像素
            target_bg_channel = target_lab[:, :, i][background_mask].flatten()
            ref_bg_channel = reference_lab[:, :, i][background_mask].flatten()
            
            if len(target_bg_channel) == 0:
                continue
            
            # 计算直方图和CDF（背景区域）
            target_hist, target_bins = np.histogram(target_bg_channel, bins=256, range=(0, 256))
            ref_hist, ref_bins = np.histogram(ref_bg_channel, bins=256, range=(0, 256))
            
            target_cdf = np.cumsum(target_hist).astype(np.float64)
            target_cdf = target_cdf / (target_cdf[-1] + 1e-6)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-6)
            
            # 构建查找表（LUT） - 使用插值让映射更平滑
            # 创建256个值的映射
            target_values = np.linspace(0, 255, 256)
            matched_values = np.zeros(256)
            
            for j in range(256):
                # 找到目标CDF对应的参考值
                target_cdf_val = target_cdf[j] if j < len(target_cdf) else 1.0
                idx = np.searchsorted(ref_cdf, target_cdf_val)
                idx = min(idx, 255)
                matched_values[j] = idx
            
            # 使用插值应用（平滑但强力）
            channel_data = target_lab[:, :, i]
            channel_clipped = np.clip(channel_data, 0, 255)
            bg_corrected_lab[:, :, i] = np.interp(channel_clipped.flatten(), 
                                                   target_values, 
                                                   matched_values).reshape(target_lab.shape[:2])
            
            print(f"    通道{i}: 变换范围 [{target_bg_channel.min():.1f}, {target_bg_channel.max():.1f}] "
                  f"-> [{ref_bg_channel.min():.1f}, {ref_bg_channel.max():.1f}]")
        
        # 3. 应用颜色变换（根据区域权重）
        print("  创建权重图...")
        result_lab = target_lab.copy()
        
        # 创建权重图
        weight_map = background_mask.astype(np.float32)
        
        # 对前景区域的权重进行空间衰减
        foreground_mask = ~background_mask
        if foreground_mask.sum() > 0:
            # 距离变换
            dist_transform = cv2.distanceTransform(foreground_mask.astype(np.uint8), cv2.DIST_L2, 5)
            if dist_transform.max() > 0:
                dist_norm = dist_transform / dist_transform.max()
                # 使用平滑的衰减曲线
                decay = np.exp(-dist_norm[foreground_mask] * 0.8)  
                # 提高前景最低权重到 0.70（边界更平滑）
                # 边缘：0.94（接近背景）
                # 中心：0.70（保留30%原色）
                weight_map[foreground_mask] = np.clip(0.70 + decay * 0.24, 0.70, 0.94)
        
        # 平滑权重图（超大范围平滑，消除边界）
        weight_map = cv2.GaussianBlur(weight_map, (blur_size1, blur_size1), 0)
        # 二次平滑
        weight_map = cv2.GaussianBlur(weight_map, (blur_size2, blur_size2), 0)
        # 三次平滑
        weight_map = cv2.GaussianBlur(weight_map, (blur_size3, blur_size3), 0)
        # 四次平滑，极致柔化
        weight_map = cv2.GaussianBlur(weight_map, (blur_size4, blur_size4), 0)
        
        print(f"  权重范围: [{weight_map.min():.2f}, {weight_map.max():.2f}]")
        
        # 根据权重混合原图和校正后的图
        for i in range(3):
            result_lab[:, :, i] = (
                target_lab[:, :, i] * (1 - weight_map) +  # 原始值
                bg_corrected_lab[:, :, i] * weight_map  # 校正后的值
            )
        
        # 限制范围
        result_lab = np.clip(result_lab, 0, 255)
        
        # 转回 RGB
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        print("✓ 区域感知颜色对齐完成")
        
        return result
    
    def _detect_difference(self, img1, img2):
        """
        检测两张图的差异
        
        返回差异图（值越大表示差异越大）
        """
        # 转灰度
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # 归一化
        gray1 = gray1 / 255.0
        gray2 = gray2 / 255.0
        
        # 计算结构相似度的逆（差异）
        # 使用局部标准差的差异
        # 自适应模糊大小（约2%的图像尺寸）
        img_size = min(img1.shape[:2])
        blur_size = max(int(img_size * 0.02), 11)
        if blur_size % 2 == 0:
            blur_size += 1
        
        # 局部均值
        mean1 = cv2.GaussianBlur(gray1, (blur_size, blur_size), 0)
        mean2 = cv2.GaussianBlur(gray2, (blur_size, blur_size), 0)
        
        # 局部方差
        var1 = cv2.GaussianBlur(gray1**2, (blur_size, blur_size), 0) - mean1**2
        var2 = cv2.GaussianBlur(gray2**2, (blur_size, blur_size), 0) - mean2**2
        
        # 差异度量（考虑均值和方差的差异）
        mean_diff = np.abs(mean1 - mean2)
        var_diff = np.abs(np.sqrt(np.maximum(var1, 0)) - np.sqrt(np.maximum(var2, 0)))
        
        # 综合差异
        diff = mean_diff + var_diff * 0.5
        
        return diff
    


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "PowerfulColorMatch": ColorMatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PowerfulColorMatch": "Powerful Color Match"
}
