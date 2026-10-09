"""Dependency-free graphical setup; the deployment process remains the source of truth."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_progress(line):
    marker = 'DEPLOY_EVENT '
    if marker not in line:
        return None
    try:
        event = json.loads(line.split(marker, 1)[1])
        if not isinstance(event, dict) or not isinstance(event.get('message'), str):
            return None
        step, total = event['step'], event['total']
        if not isinstance(step, int) or not isinstance(total, int) or not 0 <= step <= total or total <= 0:
            return None
        return event
    except (ValueError, KeyError):
        return None


def main():
    import tkinter as tk
    from tkinter import messagebox, ttk
    window = tk.Tk()
    window.title('声音克隆 · 安装与启动')
    window.geometry('760x510')
    window.minsize(620, 450)
    window.configure(background='#f3f5fa')
    style = ttk.Style(window)
    style.theme_use('clam')
    style.configure('TFrame', background='#f3f5fa')
    style.configure('TLabel', background='#f3f5fa', foreground='#24324b', font=('Microsoft YaHei UI', 10))
    style.configure('Title.TLabel', font=('Microsoft YaHei UI', 21, 'bold'))
    style.configure('Status.TLabel', font=('Microsoft YaHei UI', 12, 'bold'))
    style.configure('TButton', font=('Microsoft YaHei UI', 10), padding=8)
    style.configure('Horizontal.TProgressbar', background='#5264e8', troughcolor='#e2e7f0')
    content = ttk.Frame(window, padding=28)
    content.pack(fill='both', expand=True)
    ttk.Label(content, text='声音克隆', style='Title.TLabel').pack(anchor='w')
    ttk.Label(content, text='首次自动准备环境和模型，以后直接启动桌面窗口。').pack(anchor='w', pady=(6, 18))
    status = tk.StringVar(value='正在读取配置…')
    ttk.Label(content, textvariable=status, style='Status.TLabel', wraplength=680).pack(anchor='w')
    bar = ttk.Progressbar(content, maximum=8, mode='determinate')
    bar.pack(fill='x', pady=(12, 8))
    elapsed = tk.StringVar(value='模型与依赖首次下载可能需要较长时间，请保持网络连接。')
    ttk.Label(content, textvariable=elapsed, wraplength=680).pack(anchor='w')
    details = tk.Text(content, height=10, font=('Consolas', 9), background='white', foreground='#475569',
                      relief='flat', padx=10, pady=10, state='disabled')
    details.pack(fill='both', expand=True, pady=(16, 12))
    actions = ttk.Frame(content)
    actions.pack(fill='x')
    ttk.Button(actions, text='打开项目文件夹', command=lambda: os.startfile(ROOT)).pack(side='left')
    ttk.Button(actions, text='查看部署日志', command=lambda: os.startfile(ROOT / 'logs') if (ROOT / 'logs').is_dir() else None).pack(side='left', padx=8)
    messages = queue.Queue()
    process = None
    finished = False
    started = time.monotonic()

    def append(line):
        details.configure(state='normal')
        details.insert('end', line + '\n')
        if int(details.index('end-1c').split('.')[0]) > 800:
            details.delete('1.0', '100.0')
        details.see('end')
        details.configure(state='disabled')

    def run_setup():
        nonlocal process
        try:
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            env['PYTHONUNBUFFERED'] = '1'
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'deploy.py'), *sys.argv[1:]],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', creationflags=subprocess.CREATE_NO_WINDOW)
            for line in process.stdout:
                messages.put(('line', line.rstrip()))
            process.stdout.close()
            messages.put(('exit', process.wait()))
        except Exception as exc:
            messages.put(('line', str(exc)))
            messages.put(('exit', 1))

    def stop():
        nonlocal finished
        if finished:
            window.destroy()
            return
        if not messagebox.askyesno('暂停准备', '暂停后会保留已完成的文件。下次双击快速部署即可继续。\n现在暂停吗？', parent=window):
            return
        if process is not None and process.poll() is None:
            try:
                stopped = subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                    capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                if stopped.returncode and process.poll() is None:
                    process.kill()
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        finished = True
        window.destroy()

    def retry():
        nonlocal finished, started
        if not finished:
            return
        finished = False
        started = time.monotonic()
        status.set('正在重新检查并继续准备…')
        retry_button.pack_forget()
        close_button.configure(text='暂停')
        threading.Thread(target=run_setup, daemon=True).start()

    close_button = ttk.Button(actions, text='暂停', command=stop)
    close_button.pack(side='right')
    retry_button = ttk.Button(actions, text='重试', command=retry)
    window.protocol('WM_DELETE_WINDOW', stop)

    def poll():
        nonlocal finished
        for _ in range(200):
            try:
                kind, value = messages.get_nowait()
            except queue.Empty:
                break
            if kind == 'line':
                event = parse_progress(value)
                if event:
                    status.set(event['message'])
                    bar['maximum'] = event['total']
                    bar['value'] = event['step']
                else:
                    append(value)
            elif kind == 'exit':
                finished = True
                close_button.configure(text='关闭')
                if value == 0:
                    status.set('准备完成，桌面窗口已启动。')
                    bar['value'] = bar['maximum']
                    window.after(1000, window.destroy)
                    return
                status.set('准备未完成，请查看下方原因。处理后可点击重试。')
                retry_button.pack(side='right', padx=8)
        if not finished:
            seconds = int(time.monotonic() - started)
            elapsed.set(f'已用时 {seconds // 60:02d}:{seconds % 60:02d} · 下载中断可继续 · 显卡按实际推理判断')
        window.after(100, poll)

    threading.Thread(target=run_setup, daemon=True).start()
    window.after(100, poll)
    window.mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # Tcl/Tk may be absent in a custom Python; retain a visible console fallback.
        import traceback
        (ROOT / 'logs').mkdir(exist_ok=True)
        (ROOT / 'logs/setup-window-error.log').write_text(traceback.format_exc(), encoding='utf-8')
        subprocess.Popen(['cmd.exe', '/c', str(ROOT / 'One_Click_Start.bat'), '--console'], cwd=ROOT)
