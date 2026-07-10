"""
本地 Playwright 渲染器

优先使用本机已安装的 Playwright/Chromium 直接渲染 HTML → PNG，
无需依赖远程 T2I 服务器。若本机没有安装 Playwright，自动回退到
框架提供的 html_render（远程 T2I 服务）。

使用方式（在 main.py 里）：
    from src.infrastructure.reporting.local_renderer import make_render_func
    render_func = make_render_func(self.html_render)
    # 之后把 render_func 传给 AutoScheduler / _send_analysis_report
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable

from ...utils.logger import logger

# ──────────────────────────────────────────────
# 可用性检测（模块加载时只跑一次）
# ──────────────────────────────────────────────

def _check_playwright() -> tuple[bool, str]:
    """
    检查 playwright Python 包是否可以导入，以及 Chromium 是否已安装。
    注意：该函数可能在 AstrBot 的 asyncio 事件循环中执行，不能使用
    playwright.sync_api，否则会误报 "using Playwright Sync API inside the asyncio loop"。
    真正启动浏览器放到异步渲染路径中完成。
    返回 (可用, 原因描述)。
    """
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return False, "playwright 包未安装"

    try:
        import os
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "playwright", "install", "--dry-run", "chromium"],
            capture_output=True, text=True, timeout=15
        )
        output = (result.stdout + result.stderr).lower()
        # 新版 Playwright dry-run 在已安装/可安装时通常返回 0；缺依赖或未安装会在运行时由 async launch 给出明确错误。
        if result.returncode == 0:
            return True, "Chromium 可用（async 检测）"
        if "chromium" in output and ("browser" in output or "install" in output):
            return True, "Chromium 可用（dry-run）"
        return False, f"Chromium 未安装或检测失败: {output[-200:]}"
    except Exception as e:
        return False, f"检测失败: {e}"


# 模块加载时检测一次，结果缓存
_PLAYWRIGHT_AVAILABLE, _PLAYWRIGHT_REASON = _check_playwright()
logger.info(
    f"[LocalRenderer] Playwright 检测: "
    f"{'✓ 可用' if _PLAYWRIGHT_AVAILABLE else '✗ 不可用'} — {_PLAYWRIGHT_REASON}"
)


# ──────────────────────────────────────────────
# 浏览器上下文池（避免每次渲染都冷启动）
# ──────────────────────────────────────────────

class _BrowserPool:
    """单例浏览器池，复用 browser 实例减少渲染延迟。"""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._lock = asyncio.Lock()

    async def _ensure_started(self):
        if self._browser and self._browser.is_connected():
            return
        from playwright.async_api import async_playwright
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        logger.debug("[LocalRenderer] Chromium 浏览器已启动")

    async def screenshot(self, html_content: str, image_options: dict) -> bytes:
        async with self._lock:
            await self._ensure_started()

        # 每次新建 context + page，保证隔离
        scale_level = image_options.get("device_scale_factor_level")
        scale_map = {"normal": 1.0, "high": 1.3, "ultra": 1.8}
        device_scale_factor = image_options.get("device_scale_factor", scale_map.get(scale_level, 2))
        context = await self._browser.new_context(
            viewport={
                "width": image_options.get("viewport_width", 1000),
                "height": image_options.get("viewport_height", 600),
            },
            device_scale_factor=device_scale_factor,
        )
        page = await context.new_page()
        try:
            await page.set_content(html_content, wait_until="networkidle", timeout=20000)
            img_bytes = await page.screenshot(
                full_page=image_options.get("full_page", True),
                type=image_options.get("type", "png"),
                quality=image_options.get("quality") if image_options.get("type") == "jpeg" else None,
            )
            return img_bytes
        finally:
            await page.close()
            await context.close()

    async def close(self):
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


_pool = _BrowserPool() if _PLAYWRIGHT_AVAILABLE else None


# ──────────────────────────────────────────────
# 本地渲染入口
# ──────────────────────────────────────────────

async def _local_render(
    html_content: str,
    _data: dict,
    _return_url: bool,
    image_options: dict,
) -> bytes | None:
    """
    与框架 html_render 签名兼容的本地渲染函数。
    直接用本机 Playwright 截图，返回 PNG/JPEG bytes。
    """
    try:
        img_bytes = await _pool.screenshot(html_content, image_options)
        # QQ 对超宽/超大图兼容差；本地渲染后统一压到安全宽度。
        try:
            from PIL import Image
            im = Image.open(BytesIO(img_bytes))
            w, h = im.size
            max_w = int(image_options.get("max_output_width", 2000))
            if w > max_w:
                new_h = max(1, int(h * max_w / w))
                im = im.resize((max_w, new_h), Image.LANCZOS)
                buf = BytesIO()
                fmt = "JPEG" if image_options.get("type") == "jpeg" else "PNG"
                if fmt == "JPEG" and im.mode in ("RGBA", "LA", "P"):
                    im = im.convert("RGB")
                save_kwargs = {"quality": image_options.get("quality", 80)} if fmt == "JPEG" else {}
                im.save(buf, format=fmt, **save_kwargs)
                img_bytes = buf.getvalue()
                logger.info(f"[LocalRenderer] 图片宽度 {w}px 过大，已缩放到 {max_w}x{new_h}")
        except Exception as e:
            logger.warning(f"[LocalRenderer] 图片尺寸压缩跳过: {e}")
        logger.debug(f"[LocalRenderer] 渲染完成，图片大小: {len(img_bytes)} bytes")
        return img_bytes
    except Exception as e:
        logger.error(f"[LocalRenderer] 本地渲染失败: {e}", exc_info=True)
        return None


# ──────────────────────────────────────────────
# 公共工厂函数
# ──────────────────────────────────────────────

def make_render_func(framework_html_render: Callable) -> Callable:
    """
    返回一个渲染函数：
    - 当前配置为强制使用远程 T2I 服务，避免本机 Playwright 截图超时
    - 远程 T2I endpoint 在 /AstrBot/data/cmd_config.json 的 t2i_endpoint 配置
    """
    if _PLAYWRIGHT_AVAILABLE and _pool is not None:
        logger.info("[LocalRenderer] 使用本地 Playwright 渲染，避免远程 T2I 超宽/不稳定问题")
        return _local_render
    logger.warning(f"[LocalRenderer] 本地 Playwright 不可用，回退远程 T2I: {_PLAYWRIGHT_REASON}")
    return framework_html_render


async def shutdown():
    """插件卸载时调用，释放浏览器进程。"""
    if _pool:
        await _pool.close()
        logger.info("[LocalRenderer] Chromium 浏览器已关闭")
