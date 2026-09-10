from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
import re
import os

def generate_blank_pdf():
    reader = PdfReader('templates/Land_Acquisition_Project_Data_Sheet_DEMO.pdf')
    writer = PdfWriter()
    
    def is_filled_block(p_num, x, top, text):
        if p_num == 1:
            # Project ID boxes (x in 160-540, top ~ 211.5)
            if abs(top - 211.5) < 3.0 and x > 160.0:
                return True
            # Sector checkbox 'X' at [166.1, 248.9]
            if abs(top - 248.9) < 4.0 and abs(x - 166.1) < 10.0:
                return True
            # State boxes (x in 160-470, top ~ 303.3)
            if abs(top - 303.3) < 3.0 and x > 160.0:
                return True
            # District boxes (x in 160-470, top ~ 334.5)
            if abs(top - 334.5) < 3.0 and x > 160.0:
                return True
            # Terrain checkbox 'X' at [273.8, 371.9]
            if abs(top - 371.9) < 4.0 and abs(x - 273.8) < 10.0:
                return True
            # Latitude & Longitude boxes (top ~ 473.7)
            if abs(top - 473.7) < 3.0 and (140.0 < x < 280.0 or x > 360.0):
                return True
            # Road connectivity checkbox 'X' at [302.2, 511.1]
            if abs(top - 511.1) < 4.0 and abs(x - 302.2) < 10.0:
                return True
            # Site address text (top ~ 569.2, x > 180.0)
            if abs(top - 569.2) < 3.0 and x > 180.0:
                return True
        elif p_num == 2:
            # Cost boxes (top ~ 183.1, x > 195.0)
            if abs(top - 183.1) < 3.0 and x > 195.0:
                return True
            # Land extent boxes (top ~ 214.3, x > 195.0)
            if abs(top - 214.3) < 3.0 and x > 195.0:
                return True
            # SIA checkbox 'X' at [285.2, 303.3]
            if abs(top - 303.3) < 4.0 and abs(x - 285.2) < 10.0:
                return True
            # Forest checkbox 'X' at [200.1, 364.0]
            if abs(top - 364.0) < 4.0 and abs(x - 200.1) < 10.0:
                return True
            # Fund disbursed boxes (top ~ 422.6, x > 195.0)
            if abs(top - 422.6) < 3.0 and x > 195.0:
                return True
            # Sec 11 days boxes (top ~ 505.4, x > 240.0)
            if abs(top - 505.4) < 3.0 and x > 240.0:
                return True
            # Sec 26 mult boxes (top ~ 536.6, x > 240.0)
            if abs(top - 536.6) < 3.0 and x > 240.0:
                return True
            # PAFs count boxes (top ~ 567.8, x > 240.0)
            if abs(top - 567.8) < 3.0 and x > 240.0:
                return True
            # Dispute rate boxes (top ~ 599.0, x > 240.0)
            if abs(top - 599.0) < 3.0 and x > 240.0:
                return True
            # Protest flag checkbox 'X' at [296.5, 636.4]
            if abs(top - 636.4) < 4.0 and abs(x - 296.5) < 10.0:
                return True
            # Officer Name / Date at bottom (top ~ 718.0, x > 180.0)
            if abs(top - 718.0) < 3.0 and x > 180.0:
                return True
            # Designation (top ~ 743.5, x > 180.0)
            if abs(top - 743.5) < 3.0 and x > 180.0:
                return True
        return False

    for p_idx, page in enumerate(reader.pages):
        p_num = p_idx + 1
        stream = page.get_contents().get_data().decode('latin1')
        
        # Match each BT ... ET block
        def filter_block(match):
            block = match.group(0)
            tm = re.search(r'1 0 0 1 ([\d\.]+) ([\d\.]+) Tm', block)
            txt = ''.join(re.findall(r'\((.*?)\)', block))
            if tm:
                x, y = float(tm.group(1)), float(tm.group(2))
                top = 841.89 - y
                if is_filled_block(p_num, x, top, txt):
                    return ""
            return block

        new_stream = re.sub(r'BT\s*.*?\s*ET', filter_block, stream, flags=re.DOTALL)
        
        d = DecodedStreamObject()
        d.set_data(new_stream.encode('latin1'))
        page[NameObject("/Contents")] = d
        writer.add_page(page)

    os.makedirs('templates', exist_ok=True)
    os.makedirs('dashboard/templates', exist_ok=True)
    
    out_paths = [
        'templates/Form_LA-7_Blank_Template.pdf',
        'dashboard/templates/Form_LA-7_Blank_Template.pdf'
    ]
    for p in out_paths:
        with open(p, 'wb') as f:
            writer.write(f)
        print(f"Generated blank template at: {p}")

if __name__ == '__main__':
    generate_blank_pdf()
