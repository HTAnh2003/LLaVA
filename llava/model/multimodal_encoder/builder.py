import os
from .clip_encoder import CLIPVisionTower
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # is_absolute_path_exists = os.path.exists(image_tower)
    if vision_tower .startswith("openai") or vision_tower .startswith("laion"):
        return CLIPVisionTower(vision_tower , args=vision_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if image_tower.startswith("google"):
        return SiglipVisionTower(vision_tower , args=vision_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(vision_tower, args=vision_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {vision_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================