"""Generate the SVG and PNG call-flow reference from one declarative layout."""
import html
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WIDTH, HEIGHT = 1600, 1550
COLORS = {'ink':'#24324b','muted':'#68768e','line':'#5264e8','background':'#f3f5fa',
          'blue':'#edf0ff','green':'#e8f5ef','amber':'#fff5e4'}


def build():
    canvas=Image.new('RGB',(WIDTH,HEIGHT),COLORS['background'])
    draw=ImageDraw.Draw(canvas)
    font_root=Path(os.environ.get('WINDIR',r'C:\Windows'))/'Fonts'
    normal_path=font_root/'msyh.ttc'
    bold_path=font_root/'msyhbd.ttc'
    if not normal_path.exists():
        raise RuntimeError('Diagram export needs a Chinese font (Microsoft YaHei). SVG can be viewed without regeneration.')
    fonts={}
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
         '<title>声音克隆 0.2.0 代码调用结构</title>',
         '<desc>一键部署、桌面 JSON 通信、后台模型推理与音频保存；实线执行，虚线数据返回。</desc>',
         '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#5264e8"/></marker></defs>',
         f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["background"]}"/>']

    def font(size,bold=False):
        key=(size,bold)
        if key not in fonts:
            fonts[key]=ImageFont.truetype(str(bold_path if bold and bold_path.exists() else normal_path),size)
        return fonts[key]

    def text(x,y,content,size=24,bold=False,color=None,center=False):
        color=color or COLORS['ink']
        draw.text((x,y),content,font=font(size,bold),fill=color,anchor='mt' if center else 'lt')
        svg.append(f'<text x="{x}" y="{y+size}" text-anchor="{"middle" if center else "start"}" font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(content)}</text>')

    def panel(x,y,width,height,title,rows=(),fill='white',title_size=27):
        draw.rounded_rectangle((x,y,x+width,y+height),radius=18,fill=fill,outline='#dce3ef',width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{fill}" stroke="#dce3ef" stroke-width="2"/>')
        text(x+22,y+18,title,title_size,True)
        for index,row in enumerate(rows):
            text(x+22,y+60+index*33,row,23,color=COLORS['muted'])

    def arrow(points,label=None,dashed=False,label_xy=None):
        color=COLORS['line']
        if dashed:
            import math
            for first,last in zip(points,points[1:]):
                length=math.dist(first,last)
                for start in range(0,int(length),18):
                    end=min(start+10,length)
                    if not length: continue
                    a=tuple(first[i]+(last[i]-first[i])*start/length for i in range(2))
                    b=tuple(first[i]+(last[i]-first[i])*end/length for i in range(2))
                    draw.line((a,b),fill=color,width=3)
        else:
            draw.line(points,fill=color,width=3,joint='curve')
        end,previous=points[-1],points[-2]
        if abs(end[0]-previous[0])>abs(end[1]-previous[1]):
            direction=1 if end[0]>previous[0] else -1
            triangle=[end,(end[0]-direction*13,end[1]-7),(end[0]-direction*13,end[1]+7)]
        else:
            direction=1 if end[1]>previous[1] else -1
            triangle=[end,(end[0]-7,end[1]-direction*13),(end[0]+7,end[1]-direction*13)]
        draw.polygon(triangle,fill=color)
        coords=' '.join(f'{x},{y}' for x,y in points)
        dash=' stroke-dasharray="10 8"' if dashed else ''
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"{dash} marker-end="url(#arrow)"/>')
        if label and label_xy:
            text(*label_xy,label,22,color=color)

    text(60,38,'声音克隆 · 代码调用结构',42,True)
    text(60,96,'Voice Clone 0.2.0    实线：主要执行流程    虚线：数据 / 状态返回',24,color=COLORS['muted'])
    text(60,165,'01  一键部署',30,True)
    panel(60,215,235,115,'快速部署.bat',['中文双击入口'],COLORS['blue'],26)
    panel(340,215,335,115,'One_Click_Start.bat',['uv → 独立 Python'],COLORS['blue'],26)
    panel(720,215,325,115,'setup_app.py',['main()  安装窗口'],COLORS['blue'])
    panel(1090,215,450,115,'deploy.py main()',['配置 / 硬件 / 部署锁'],COLORS['blue'])
    arrow([(295,270),(340,270)])
    arrow([(675,270),(720,270)])
    arrow([(1045,270),(1090,270)])
    panel(60,390,985,125,'Deployer.deploy() / launch()',[
        'Python / venv → 固定依赖 → CUDA 实测 → 模型下载校验',
        '音频工具 → 短语音验证 → 桌面快捷方式 → 启动 GUI'],COLORS['green'])
    panel(1090,390,450,125,'配置输入',[
        'deploy_config.json / requirements.lock',
        '可选 deploy.local.json（不随源码发布）'],title_size=27)
    arrow([(1315,330),(1315,355),(552,355),(552,390)])
    arrow([(1315,390),(1315,330)],'读取配置',label_xy=(1340,345))
    text(60,565,'02  桌面界面与后台通信',30,True)
    panel(60,620,400,170,'desktop_app.py',[
        'main() → VoiceWindow',
        '录音 / 文字 / 试听 / 历史',
        'read_worker() → set_result()'],COLORS['blue'])
    panel(505,620,420,170,'VoiceWindow.generate()',[
        '校验输入 / 保存设置',
        '启动 QProcess 后台',
        'send_request() 写 UTF-8 JSON'],COLORS['blue'],26)
    panel(970,620,570,170,'desktop_worker.py',[
        'main() 读取 stdin 的 JSON 行',
        'InferenceEngine.generate(request, event)',
        'event() 回传 stage / done / error'],COLORS['blue'])
    arrow([(552,515),(552,595),(260,595),(260,620)])
    arrow([(460,705),(505,705)])
    arrow([(925,705),(970,705)])
    arrow([(1255,790),(1255,835),(260,835),(260,790)],
          'VOICE_EVENT → 阶段 / 文件路径 / 用时 / 错误',True,(455,800))
    text(60,880,'取消 / 关闭 / 空闲计时 → stop_worker() → 停止进程树，释放模型显存',24,color=COLORS['muted'])
    text(60,945,'03  后台推理与保存',30,True)
    panel(60,1005,1480,370,'voice_clone.inference.InferenceEngine.generate()',[
        '按需缓存模型；split_text() 分段；复用参考声音提示；拼接全部段落后保存 WAV'],COLORS['amber'],29)
    panel(90,1120,420,210,'audio.py',[
        'reference_info()',
        'load_reference()',
        '校验 / 解码 / 转为单声道'],title_size=27)
    panel(555,1120,505,210,'Qwen3TTSModel',[
        'from_pretrained() 按需加载',
        'create_voice_clone_prompt()',
        'generate_voice_clone() 逐段生成'],title_size=27)
    panel(1105,1120,405,210,'audio.write_result()',[
        '验证波形与采样率',
        '临时文件完成后原子替换',
        'outputs/*.wav'],title_size=27)
    arrow([(1255,790),(1255,925),(800,925),(800,1005)])
    arrow([(510,1225),(555,1225)])
    arrow([(1060,1225),(1105,1225)])
    arrow([(1540,1080),(1570,1080),(1570,705),(1540,705)],dashed=True)
    text(60,1430,'辅助服务：project.py（路径 / 原子设置）  ·  diagnostics.py（诊断报告）',25,color=COLORS['muted'])
    text(60,1475,'兼容入口：app.py（Gradio 网页）  ·  scripts / launchers（旧辅助启动）',24,color=COLORS['muted'])
    svg.append('</svg>')
    assets=ROOT/'assets'
    assets.mkdir(exist_ok=True)
    (assets/'代码调用结构.svg').write_text('\n'.join(svg),encoding='utf-8',newline='\n')
    canvas.save(assets/'代码调用结构.png',optimize=True)
    print('SVG and PNG code call-flow diagrams generated.')


if __name__=='__main__':
    build()
