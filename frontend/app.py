import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

BACKEND_URL = "http://localhost:8000/predict"

# ===============================
# Backend Call
# ===============================
def analyze_email(subject, body, history):
    if not subject or not body:
        raise gr.Error("Subject and Body are required")

    payload = {"subject": subject, "body": body}
    response = requests.post(BACKEND_URL, json=payload)

    if response.status_code != 200:
        raise gr.Error("FastAPI backend not reachable")

    result = response.json()

    record = {
        "subject": subject,
        "body": body,
        "category": result["category"],
        "category_confidence": result["category_confidence"],
        "urgency": result["urgency"],
        "urgency_confidence": result["urgency_confidence"],
        "timestamp": result["timestamp"]
    }

    history.append(record)

    ui_result = f"""
### 📧 Email Analysis Result

**Category:** `{result['category']}`  
**Category Confidence:** `{result['category_confidence'] * 100:.2f}%`

**Urgency Level:** `{result['urgency']}`  
**Urgency Confidence:** `{result['urgency_confidence'] * 100:.2f}%`

**Analyzed At:** `{result['timestamp']}`
"""

    return gr.Markdown(ui_result), history


# ===============================
# Analytics with Slicers
# ===============================
def generate_analytics(history, category_filter, urgency_filter, date_range):
    if not history:
        return "No data available", None, None, None

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -----------------------------
    # Apply slicers
    # -----------------------------
    if category_filter != "All":
        df = df[df["category"] == category_filter]

    if urgency_filter != "All":
        df = df[df["urgency"] == urgency_filter]

    if date_range:
        start, end = date_range
        df = df[
            (df["timestamp"].dt.date >= start) &
            (df["timestamp"].dt.date <= end)
        ]

    if df.empty:
        return "No data after applying filters", None, None, None

    # -----------------------------
    # KPI Calculations
    # -----------------------------
    total_emails = len(df)
    top_category = df["category"].mode()[0]
    high_urgency_pct = round(
        (df["urgency"].value_counts().get("High", 0) / total_emails) * 100, 2
    )

    kpi_text = f"""
## 📊 Key Metrics

🔹 **Total Emails:** `{total_emails}`  
🔹 **Top Category:** `{top_category}`  
🔹 **High Urgency %:** `{high_urgency_pct}%`
"""

    # -----------------------------
    # Category Bar Chart
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    df["category"].value_counts().plot(
        kind="bar",
        color="#0b5ed7",
        ax=ax1
    )
    ax1.set_title("Email Category Distribution")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Count")

    # -----------------------------
    # Urgency Donut Chart
    # -----------------------------
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    urgency_counts = df["urgency"].value_counts()

    ax2.pie(
        urgency_counts,
        labels=urgency_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#cfe2ff", "#6ea8fe", "#084298"],
        wedgeprops=dict(width=0.4)
    )
    ax2.set_title("Urgency Level Distribution")

    # -----------------------------
    # Trend Line Chart
    # -----------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    df.groupby(df["timestamp"].dt.date).size().plot(
        kind="line",
        marker="o",
        color="#0b5ed7",
        ax=ax3
    )
    ax3.set_title("Email Volume Over Time")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Email Count")

    return kpi_text, fig1, fig2, fig3


# ===============================
# Downloads
# ===============================
def download_csv(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    path = "email_results.csv"
    df.to_csv(path, index=False)
    return path


def download_pdf(history):
    if not history:
        return None

    path = "email_report.pdf"
    c = canvas.Canvas(path, pagesize=A4)
    text = c.beginText(40, 800)
    text.setFont("Helvetica", 10)

    text.textLine("EmailSort – Analytics Report")
    text.textLine("-" * 70)

    for i, record in enumerate(history, start=1):
        text.textLine(f"{i}. Subject: {record['subject']}")
        text.textLine(f"   Category: {record['category']} ({record['category_confidence']})")
        text.textLine(f"   Urgency: {record['urgency']} ({record['urgency_confidence']})")
        text.textLine(f"   Timestamp: {record['timestamp']}")
        text.textLine("")

    c.drawText(text)
    c.save()
    return path


# ===============================
# UI
# ===============================
with gr.Blocks(
    title="EmailSort AI",
    css="""
    body { background-color: #f8fbff; }
    h1, h2, h3 { color: #0b5ed7; }
    .gr-button { background-color: #0b5ed7 !important; color: white !important; }
    """
) as demo:

    gr.Markdown("# 📬 EmailSort Dashboard")
    gr.Markdown("### Enterprise Email Classification & Analytics")

    history = gr.State([])

    with gr.Tabs():

        # -------------------------
        # Analyze
        # -------------------------
        with gr.Tab("Analyze Email"):
            subject = gr.Textbox(label="Email Subject")
            body = gr.Textbox(label="Email Body", lines=6)
            analyze_btn = gr.Button("Analyze Email")

            result_output = gr.Markdown()

            analyze_btn.click(
                analyze_email,
                inputs=[subject, body, history],
                outputs=[result_output, history]
            )

        # -------------------------
        # Analytics (Power BI Style)
        # -------------------------
        with gr.Tab("Analytics"):
            gr.Markdown("## 📊 Analytics Dashboard")

            with gr.Row():
                category_filter = gr.Dropdown(
                    ["All", "Complaint", "Feedback", "Spam", "Inquiry"],
                    label="Category"
                )
                urgency_filter = gr.Dropdown(
                    ["All", "Low", "Medium", "High"],
                    label="Urgency"
                )

            refresh_btn = gr.Button("Apply Filters")

            kpi_output = gr.Markdown()

            with gr.Row():
                category_plot = gr.Plot()
                urgency_plot = gr.Plot()

            trend_plot = gr.Plot()

            refresh_btn.click(
                generate_analytics,
                inputs=[history, category_filter, urgency_filter],
                outputs=[kpi_output, category_plot, urgency_plot, trend_plot]
            )

        # -------------------------
        # Downloads
        # -------------------------
        with gr.Tab("Download"):
            gr.Markdown("### Export Results")

            csv_btn = gr.Button("Download CSV")
            pdf_btn = gr.Button("Download PDF Report")

            file_output = gr.File()

            csv_btn.click(download_csv, inputs=history, outputs=file_output)
            pdf_btn.click(download_pdf, inputs=history, outputs=file_output)


demo.launch(share=True)
