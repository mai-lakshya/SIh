import pdfplumber

def test_layouts():
    files = [
        ('Layout A (Demo)', 'templates/Land_Acquisition_Project_Data_Sheet_DEMO.pdf'),
        ('Layout B (Sample 5)', 'templates/project_sample_5.pdf')
    ]
    for name, path in files:
        with pdfplumber.open(path) as pdf:
            p1 = pdf.pages[0]
            words = p1.extract_words()
            id_word = next((w for w in words if 'Identifier' in w['text']), None)
            top = id_word['top'] if id_word else 0
            layout = 'stacked' if top < 180 else 'inline'
            print(f"{name}: 'Identifier' top = {top:.1f} -> detected layout: {layout.upper()}")

if __name__ == '__main__':
    test_layouts()
