#!/usr/bin/env python3
"""
RAGInsight 项目一键启动脚本
启动后端 FastAPI 服务 + 前端 Vite 开发服务器

用法:
    python start_project.py

按 Ctrl+C 优雅停止所有服务
"""

import os
import sys
import time
import signal
import subprocess
import urllib.request
from pathlib import Path

# ============ 配置 ============
PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
VENV_PYTHON = BACKEND_DIR / "venv" / "Scripts" / "python.exe"
BACKEND_PORT = 8000
FRONTEND_PORT = 5173
BACKEND_URL = f"http://localhost:{BACKEND_PORT}/health"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"
MAX_WAIT_SECONDS = 60

# 全局子进程引用
backend_proc = None
frontend_proc = None


def check_python_env():
    """检查虚拟环境是否存在"""
    if not VENV_PYTHON.exists():
        print(f"[错误] 虚拟环境未找到: {VENV_PYTHON}")
        print("请先初始化后端环境:")
        print(f"  cd backend && python -m venv venv && venv\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)
    print(f"[OK] 虚拟环境: {VENV_PYTHON}")


def wait_for_service(url: str, name: str, timeout: int = MAX_WAIT_SECONDS) -> bool:
    """轮询等待服务就绪"""
    print(f"[等待] {name} 启动中... ({url})")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[OK] {name} 已就绪 ({resp.status})")
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"[超时] {name} 在 {timeout}s 内未就绪")
    return False


def start_backend():
    """启动后端 Uvicorn 服务"""
    global backend_proc
    cmd = [
        str(VENV_PYTHON),
        "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(BACKEND_PORT),
    ]
    print(f"[启动] 后端: {' '.join(cmd)}")
    backend_proc = subprocess.Popen(
        cmd,
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def start_frontend():
    """启动前端 Vite 开发服务器"""
    global frontend_proc
    # 检查 node_modules
    if not (FRONTEND_DIR / "node_modules").exists():
        print(f"[警告] 前端 node_modules 不存在，请先运行: cd frontend && npm install")
    
    cmd = "npm run dev"
    print(f"[启动] 前端: {cmd}")
    # Windows 上使用 shell=True 确保能找到 npm
    use_shell = sys.platform == "win32"
    frontend_proc = subprocess.Popen(
        cmd,
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        shell=use_shell,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )


def stop_services():
    """优雅停止所有子进程"""
    print("\n[停止] 正在关闭服务...")
    for name, proc in [("后端", backend_proc), ("前端", frontend_proc)]:
        if proc and proc.poll() is None:
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
                print(f"[OK] {name} 已停止")
            except Exception as e:
                print(f"[警告] {name} 停止失败: {e}")
                try:
                    proc.kill()
                except Exception:
                    pass


def read_output(proc, name):
    """后台线程读取子进程输出并打印"""
    if proc is None:
        return
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [{name}] {line}")
    except Exception:
        pass


def main():
    global backend_proc, frontend_proc
    print("=" * 60)
    print("  RAGInsight 项目启动器")
    print("=" * 60)

    # 1. 检查环境
    check_python_env()

    # 2. 启动后端
    start_backend()
    time.sleep(2)

    # 3. 启动前端
    start_frontend()
    time.sleep(2)

    # 4. 等待服务就绪
    backend_ready = wait_for_service(BACKEND_URL, "后端 API", timeout=30)
    frontend_ready = wait_for_service(FRONTEND_URL, "前端页面", timeout=30)

    if not backend_ready:
        print("[失败] 后端启动失败，请检查日志")
        stop_services()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  🚀 所有服务已启动!")
    print(f"  前端界面: {FRONTEND_URL}")
    print(f"  后端 API: {BACKEND_URL}")
    print("  按 Ctrl+C 停止服务")
    print("=" * 60 + "\n")

    # 5. 保持运行并读取输出
    import threading
    t_backend = threading.Thread(target=read_output, args=(backend_proc, "BACKEND"), daemon=True)
    t_frontend = threading.Thread(target=read_output, args=(frontend_proc, "FRONTEND"), daemon=True)
    t_backend.start()
    t_frontend.start()

    # 6. 等待中断
    try:
        while True:
            # 检查子进程是否意外退出
            if backend_proc and backend_proc.poll() is not None:
                print(f"[警告] 后端进程已退出 (code={backend_proc.returncode})")
                break
            if frontend_proc and frontend_proc.poll() is not None:
                print(f"[警告] 前端进程已退出 (code={frontend_proc.returncode})")
                # 前端意外退出不影响整体，继续运行
                frontend_proc = None
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_services()
        print("[完成] 所有服务已关闭")


if __name__ == "__main__":
    main()
