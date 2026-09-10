from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
import io
import os

# Coordinates of box grids:
# Each box has width ~ 18.7 (or 15.9 for lat/lon), height ~ 17.6
# In ReportLab, origin (0, 0) is bottom-left!
# page height = 841.89
# y_pdf = 841.89 - top
# For top ~ 211.5, y_pdf = 630.4

BOX_COORDS = {
    'project_id': {'p': 1, 'x_start': 170.7, 'dx': 18.75, 'y': 630.4},
    'state': {'p': 1, 'x_start': 170.7, 'dx': 18.75, 'y': 538.6},
    'district': {'p': 1, 'x_start': 170.7, 'dx': 18.75, 'y': 507.4},
    'latitude': {'p': 1, 'x_start': 147.3, 'dx': 15.9, 'y': 368.2},
    'longitude': {'p': 1, 'x_start': 376.3, 'dx': 15.9, 'y': 368.2},
    'site_address': {'p': 1, 'x': 201.3, 'y': 272.7}, # plain text line
    'cost': {'p': 2, 'x_start': 205.4, 'dx': 18.75, 'y': 658.8},
    'land_area': {'p': 2, 'x_start': 205.4, 'dx': 18.75, 'y': 627.6},
    'fund_pct': {'p': 2, 'x_start': 205.4, 'dx': 18.75, 'y': 419.3},
    'sec11_days': {'p': 2, 'x_start': 250.8, 'dx': 18.75, 'y': 336.5},
    'comp_mult': {'p': 2, 'x_start': 250.8, 'dx': 18.75, 'y': 305.3},
    'pafs': {'p': 2, 'x_start': 250.8, 'dx': 18.75, 'y': 274.1},
    'dispute_rate': {'p': 2, 'x_start': 250.8, 'dx': 18.75, 'y': 242.9},
}

CHECKBOX_COORDS = {
    'project_type': {
        'Highway': {'p': 1, 'x': 166.1, 'y': 593.0},
        'Railway': {'p': 1, 'x': 256.8, 'y': 593.0},
        'Irrigation': {'p': 1, 'x': 347.5, 'y': 593.0},
        'Power': {'p': 1, 'x': 166.1, 'y': 562.6},
        'Urban Dev.': {'p': 1, 'x': 256.8, 'y': 562.6},
        'Other': {'p': 1, 'x': 347.5, 'y': 562.6},
    },
    'terrain_type': {
        'Plain': {'p': 1, 'x': 166.1, 'y': 470.0},
        'Rural / Agricultural': {'p': 1, 'x': 273.8, 'y': 470.0},
        'Hilly': {'p': 1, 'x': 381.5, 'y': 470.0},
        'Forest / Tribal': {'p': 1, 'x': 166.1, 'y': 439.6},
        'Urban / Congested': {'p': 1, 'x': 273.8, 'y': 439.6},
        'Coastal': {'p': 1, 'x': 381.5, 'y': 439.6},
    },
    'road_type': {
        'Auto-Detect (PMGSY)': {'p': 1, 'x': 166.1, 'y': 330.8},
        'PMGSY Connected': {'p': 1, 'x': 302.2, 'y': 330.8},
        'State Highway': {'p': 1, 'x': 166.1, 'y': 300.4},
        'Not Connected': {'p': 1, 'x': 302.2, 'y': 300.4},
    },
    'sia_approval_status': {
        'Approved': {'p': 2, 'x': 200.1, 'y': 538.6},
        'Pending': {'p': 2, 'x': 285.2, 'y': 538.6},
        'Rejected': {'p': 2, 'x': 200.1, 'y': 508.3},
        'Not Applicable': {'p': 2, 'x': 285.2, 'y': 508.3},
    },
    'forest_clearance_status': {
        'Not Required': {'p': 2, 'x': 200.1, 'y': 477.9},
        'Pending': {'p': 2, 'x': 285.2, 'y': 477.9},
        'Approved': {'p': 2, 'x': 200.1, 'y': 447.5},
        'Rejected': {'p': 2, 'x': 285.2, 'y': 447.5},
    },
    'local_protest_flag': {
        'Yes': {'p': 2, 'x': 245.5, 'y': 205.5},
        'No': {'p': 2, 'x': 296.5, 'y': 205.5},
    }
}

def generate_filled_pdf(data: dict, output_pdf_path: str):
    blank_reader = PdfReader('templates/Form_LA-7_Blank_Template.pdf')
    writer = PdfWriter()
    
    # Create overlays for page 1 and page 2
    for p_idx, base_page in enumerate(blank_reader.pages):
        p_num = p_idx + 1
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(595.28, 841.89))
        can.setFont("Helvetica-Bold", 8.5)
        
        # Fill box grids for this page
        for field, conf in BOX_COORDS.items():
            if conf['p'] == p_num and field in data and data[field] is not None:
                val_str = str(data[field])
                if field == 'site_address':
                    can.setFont("Helvetica", 8.0)
                    can.drawString(conf['x'], conf['y'], val_str)
                    can.setFont("Helvetica-Bold", 8.5)
                else:
                    x_cur = conf['x_start']
                    for ch in val_str:
                        can.drawString(x_cur, conf['y'], ch)
                        x_cur += conf['dx']
        
        # Fill checkboxes for this page
        can.setFont("Helvetica-Bold", 8.0)
        for group_name, opt_dict in CHECKBOX_COORDS.items():
            selected_opt = data.get(group_name)
            if selected_opt in opt_dict:
                opt_info = opt_dict[selected_opt]
                if opt_info['p'] == p_num:
                    can.drawString(opt_info['x'], opt_info['y'], "X")
        
        can.save()
        packet.seek(0)
        overlay_reader = PdfReader(packet)
        base_page.merge_page(overlay_reader.pages[0])
        writer.add_page(base_page)
        
    with open(output_pdf_path, 'wb') as f:
        writer.write(f)
    print(f"Generated test PDF: {output_pdf_path}")

if __name__ == '__main__':
    # 1. Generate VARIANT PDF (e.g., Railway project in Maharashtra / Pune)
    variant_data = {
        'project_id': 'MOR-MH-2024-0042',
        'project_type': 'Railway',
        'state': 'MAHARASHTRA',
        'district': 'PUNE',
        'terrain_type': 'Hilly',
        'latitude': '18.5204',
        'longitude': '73.8567',
        'road_type': 'State Highway',
        'site_address': 'Hadapsar Junction, Pune District, Maharashtra',
        'cost': '2840.50',
        'land_area': '412.8',
        'sia_approval_status': 'Approved',
        'forest_clearance_status': 'Pending',
        'fund_pct': '68.50',
        'sec11_days': '180',
        'comp_mult': '2.10',
        'pafs': '1240',
        'dispute_rate': '8.45',
        'local_protest_flag': 'Yes'
    }
    generate_filled_pdf(variant_data, 'templates/Land_Acquisition_Project_Data_Sheet_VARIANT.pdf')
    
    # 2. Generate PARTIAL PDF (some fields deliberately omitted)
    partial_data = {
        'project_id': 'NHAI-GJ-2025-0010',
        'project_type': 'Highway',
        'state': 'GUJARAT',
        # district omitted!
        'terrain_type': 'Plain',
        'latitude': '22.2587',
        'longitude': '71.1924',
        # road_type omitted!
        'site_address': 'Rajkot Bypass, Gujarat',
        'cost': '950.00',
        # land_area omitted!
        'sia_approval_status': 'Pending',
        'forest_clearance_status': 'Not Required',
        # fund_pct omitted!
        'sec11_days': '95',
        'comp_mult': '1.50',
        'pafs': '320',
        # dispute_rate omitted!
        'local_protest_flag': 'No'
    }
    generate_filled_pdf(partial_data, 'templates/Land_Acquisition_Project_Data_Sheet_PARTIAL.pdf')
    
    # 3. Generate UNRELATED PDF (not Form LA-7 at all)
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(595.28, 841.89))
    can.drawString(100, 700, "Quarterly Financial Performance Report")
    can.drawString(100, 680, "This is an internal memo for administrative purposes only.")
    can.save()
    packet.seek(0)
    unrelated_writer = PdfWriter()
    unrelated_writer.add_page(PdfReader(packet).pages[0])
    with open('templates/unrelated_document.pdf', 'wb') as f:
        unrelated_writer.write(f)
    print("Generated unrelated PDF: templates/unrelated_document.pdf")
