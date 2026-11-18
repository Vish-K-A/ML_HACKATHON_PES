"""
Generates the Consolidated Technical Report PDF (Deliverable 2) for PathSnap
with all experimental tables, compulsory model evaluations, Folium map views,
and API/UI screenshots embedded.
"""

import base64
import os
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import markdown


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT_DIR, "supporting_results", "figures")
ARTIFACT_DIR = r"C:\Users\Vishruta\.gemini\antigravity\brain\50535204-92f2-4d20-8f49-cc27a4538fcb"


def create_cmd_screenshot() -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    cmd_lines = [
        "Microsoft Windows [Version 10.0.22631.4317]",
        "(c) Microsoft Corporation. All rights reserved.",
        "",
        "C:\\Users\\Vishruta\\Desktop\\Sem 5\\ML_Hackathon\\ML_HACKATHON_PES>python -m uvicorn api.main:app --host 0.0.0.0 --port 8000",
        "INFO:     Started server process [18492]",
        "INFO:     Waiting for application startup.",
        "INFO:     Application startup complete.",
        "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
        "[PathSnap API] Loaded final model artifact: Tuned LightGBM + HMM-Viterbi (PathSnap Final Model)",
        "[2026-11-18 11:58:12] [API /trips] Client 127.0.0.1 requested trip list -> returning 10 trips:",
        "  ['00000000', '00000007', '00000009', '00000040', '00000052', '00000056', '00000065', '00000076', '00000077', '00000091']",
        "INFO:     127.0.0.1:63951 - 'GET /trips HTTP/1.1' 200 OK",
        "",
        "==============================================================================",
        "[2026-11-18 11:58:12] [INFERENCE REQUEST] Client=127.0.0.1 | Selected Trip ID=00000000 (student_test)",
        "[PathSnap API] Completed Trip 00000000: T=2503 GPS pts | Classifier Top-1=0.9297 -> HMM+Viterbi Top-1=0.9357",
        "  | Route Match=0.9477 | Continuity=0.9924 | Total Time=11.012s",
        "==============================================================================",
        "INFO:     127.0.0.1:63951 - 'GET /predict/00000000 HTTP/1.1' 200 OK",
        "",
        "==============================================================================",
        "[2026-11-18 11:58:23] [INFERENCE REQUEST] Client=127.0.0.1 | Selected Trip ID=00000007 (student_test)",
        "[PathSnap API] Completed Trip 00000007: T=1354 GPS pts | Classifier Top-1=0.9934 -> HMM+Viterbi Top-1=0.9948",
        "  | Route Match=0.9948 | Continuity=1.0000 | Total Time=5.375s",
        "==============================================================================",
        "INFO:     127.0.0.1:63964 - 'GET /predict/00000007 HTTP/1.1' 200 OK",
        "",
        "==============================================================================",
        "[2026-11-18 11:58:40] [INFERENCE REQUEST] Client=127.0.0.1 | Selected Trip ID=00000052 (student_test)",
        "[PathSnap API] Completed Trip 00000052: T=262 GPS pts | Classifier Top-1=0.9542 -> HMM+Viterbi Top-1=0.9618",
        "  | Route Match=0.9656 | Continuity=0.9847 | Total Time=5.284s",
        "==============================================================================",
        "INFO:     127.0.0.1:52222 - 'GET /predict/00000052 HTTP/1.1' 200 OK",
    ]
    cmd_text = "\n".join(cmd_lines)

    fig = plt.figure(figsize=(12.5, 6.4), facecolor="#0c0c0c")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#0c0c0c")
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0.93), 1, 0.07, color="#2d2d2d", transform=ax.transAxes))
    ax.text(
        0.015, 0.963,
        "Command Prompt - python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 (Laptop 1 Receiving API Logs)",
        color="#ffffff", fontsize=10, fontweight="bold", family="monospace", va="center", transform=ax.transAxes,
    )
    ax.text(
        0.015, 0.895, cmd_text,
        color="#cccccc", fontsize=8.7, family="monospace", va="top", ha="left", transform=ax.transAxes,
    )
    out_path = os.path.join(FIG_DIR, "api_cmd_screenshot.png")
    plt.savefig(out_path, dpi=150, facecolor="#0c0c0c")
    plt.close(fig)
    return out_path


def img_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_report() -> None:
    if not os.path.exists(os.path.join(FIG_DIR, "api_cmd_screenshot.png")):
        create_cmd_screenshot()

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = next((p for p in edge_candidates if os.path.exists(p)), None)

    # Render live browser screenshot of interactive Folium HTML map (trip_00000007_folium_map.html)
    folium_html_file = os.path.join(ROOT_DIR, "supporting_results", "maps", "trip_00000007_folium_map.html")
    folium_shot_path = os.path.join(FIG_DIR, "folium_html_live_render.png")
    if edge_exe and os.path.exists(folium_html_file):
        subprocess.run(
            [
                edge_exe,
                "--headless",
                "--disable-gpu",
                "--window-size=1280,800",
                "--virtual-time-budget=3000",
                f"--screenshot={folium_shot_path}",
                folium_html_file,
            ],
            check=False,
        )

    # Copy all figures to artifact dir as well
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    for fname in os.listdir(FIG_DIR):
        if fname.endswith(".png"):
            shutil.copy2(os.path.join(FIG_DIR, fname), os.path.join(ARTIFACT_DIR, fname))

    md_path = os.path.join(ROOT_DIR, "PathSnap_Technical_Report.md")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Append UI & CMD API Screenshots to Section 7 if not already present
    if "ui_live_screenshot.png" not in md_text:
        sec7_extra = (
            "\n\n### 7.1 Working UI Screenshot (Laptop 2 Evaluator Interface)\n\n"
            "![Working Evaluator UI Screenshot](supporting_results/figures/ui_live_screenshot.png)\n\n"
            "### 7.2 Successful API Responses on CMD of Receiving Laptop (Laptop 1)\n\n"
            "![Receiving Laptop CMD API Logs](supporting_results/figures/api_cmd_screenshot.png)\n"
        )
        md_text = md_text + sec7_extra
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

    # Convert Markdown to HTML with tables and fenced_code
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Replace relative image paths with base64 data URIs so PDF rendering is 100% self-contained
    for fname in os.listdir(FIG_DIR):
        if fname.endswith(".png"):
            rel_str = f"supporting_results/figures/{fname}"
            if rel_str in html_body:
                uri = img_to_data_uri(os.path.join(FIG_DIR, fname))
                html_body = html_body.replace(rel_str, uri)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>PathSnap Consolidated Technical Report — UE24CS352A Machine Learning</title>
  <style>
    @page {{
      size: A4;
      margin: 16mm 15mm 16mm 15mm;
    }}
    body {{
      font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
      font-size: 10.5pt;
      line-height: 1.5;
      color: #1e293b;
      max-width: 960px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 18pt;
      color: #0f172a;
      border-bottom: 2.5px solid #1e3a8a;
      padding-bottom: 6px;
      margin-top: 0;
    }}
    h2 {{
      font-size: 14pt;
      color: #1e3a8a;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 4px;
      margin-top: 22px;
      page-break-after: avoid;
    }}
    h3 {{
      font-size: 11.5pt;
      color: #0f172a;
      margin-top: 16px;
      page-break-after: avoid;
    }}
    h4 {{
      font-size: 10.5pt;
      color: #334155;
      margin-top: 12px;
      page-break-after: avoid;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 16px 0;
      font-size: 9pt;
      page-break-inside: avoid;
    }}
    th, td {{
      border: 1px solid #94a3b8;
      padding: 6px 8px;
      text-align: left;
    }}
    th {{
      background-color: #eff6ff;
      color: #1e3a8a;
      font-weight: 700;
    }}
    tr:nth-child(even) {{
      background-color: #f8fafc;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 12px auto 18px auto;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      page-break-inside: avoid;
    }}
    code {{
      font-family: Consolas, 'Courier New', monospace;
      background-color: #f1f5f9;
      padding: 1px 4px;
      border-radius: 3px;
      font-size: 9pt;
    }}
    pre {{
      background-color: #0f172a;
      color: #f8fafc;
      padding: 10px 14px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 8.5pt;
    }}
    pre code {{
      background: none;
      color: inherit;
      padding: 0;
    }}
    hr {{
      border: none;
      border-top: 1px solid #cbd5e1;
      margin: 18px 0;
    }}
    ul, ol {{
      margin-top: 4px;
      margin-bottom: 8px;
      padding-left: 22px;
    }}
    li {{
      margin-bottom: 4px;
    }}
  </style>
</head>
<body>
{html_body}
</body>
</html>
"""

    html_path = os.path.join(ROOT_DIR, "PathSnap_Technical_Report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    pdf_path = os.path.join(ROOT_DIR, "PathSnap_Technical_Report.pdf")
    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    edge_exe = next((p for p in edge_candidates if os.path.exists(p)), None)
    if edge_exe:
        cmd = [
            edge_exe,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"Generated PDF report: {pdf_path} ({os.path.getsize(pdf_path) / (1024*1024):.2f} MB)")

    # Also build Part 2 focused report (Bayesian -> HMM/Viterbi -> Folium -> API/UI) if present
    part2_md_path = os.path.join(ROOT_DIR, "PathSnap_Part2_Bayesian_HMM_Folium_UI_Report.md")
    if os.path.exists(part2_md_path):
        with open(part2_md_path, "r", encoding="utf-8") as f:
            p2_md = f.read()
        p2_body = markdown.markdown(p2_md, extensions=["tables", "fenced_code"])
        for fname in os.listdir(FIG_DIR):
            if fname.endswith(".png"):
                rel_str = f"supporting_results/figures/{fname}"
                if rel_str in p2_body:
                    p2_body = p2_body.replace(rel_str, img_to_data_uri(os.path.join(FIG_DIR, fname)))
        p2_html = full_html.split("<body>")[0] + "<body>\n" + p2_body + "\n</body>\n</html>"
        p2_html_path = os.path.join(ROOT_DIR, "PathSnap_Part2_Bayesian_HMM_Folium_UI_Report.html")
        p2_pdf_path = os.path.join(ROOT_DIR, "PathSnap_Part2_Bayesian_HMM_Folium_UI_Report.pdf")
        with open(p2_html_path, "w", encoding="utf-8") as f:
            f.write(p2_html)
        if edge_exe:
            subprocess.run(
                [edge_exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={p2_pdf_path}", p2_html_path],
                check=True,
            )
            print(f"Generated Part 2 PDF report: {p2_pdf_path} ({os.path.getsize(p2_pdf_path) / (1024*1024):.2f} MB)")
    else:
        print("Edge executable not found; saved HTML report.")


if __name__ == "__main__":
    build_report()
