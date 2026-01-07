try:
    from fpdf import FPDF
except ImportError:
    # Dummy class to prevent NameError on class definition
    class FPDF:
        def __init__(self, *args, **kwargs):
            pass
            
    FPDF_AVAILABLE = False
else:
    FPDF_AVAILABLE = True

import tempfile
import matplotlib.pyplot as plt
import os

class ClusteringReport(FPDF):
    def header(self):
        if not FPDF_AVAILABLE: return
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Privacy-Preserving Clustering Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        if not FPDF_AVAILABLE: return
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(metrics, config, plot_image_path=None):
    """
    Generates a PDF report.
    metrics: Dict of scores
    config: Dict of system params (k, sigma, epsilon)
    plot_image_path: Path to a saved temp image of the plot
    """
    if not FPDF_AVAILABLE:
        raise ImportError("fpdf library not installed. Please install it to generate reports.")
        
    pdf = ClusteringReport()
    pdf.add_page()
    
    # Section 1: Configuration
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. System Configuration', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for k, v in config.items():
        pdf.cell(0, 7, f"{k}: {v}", 0, 1)
        
    pdf.ln(5)
    
    # Section 2: Quality Metrics
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. Clustering Quality Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if metrics:
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            pdf.cell(0, 7, f"{k}: {val_str}", 0, 1)
    else:
        pdf.cell(0, 7, "No metrics available.", 0, 1)
        
    pdf.ln(5)
    
    # Section 3: Visuals
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. Visualization', 0, 1)
    pdf.ln(5)
    
    if plot_image_path and os.path.exists(plot_image_path):
        # Center image
        # A4 width is ~210mm. Image 100mm wide?
        pdf.image(plot_image_path, x=55, w=100)
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 10, "(No visualization captured)", 0, 1)
        
    # Output to temp file
    tmp_file = tempfile.mktemp(suffix=".pdf")
    pdf.output(tmp_file)
    return tmp_file
