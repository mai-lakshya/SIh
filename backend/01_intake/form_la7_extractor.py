import io
import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
import pdfplumber

logger = logging.getLogger(__name__)

# Standard Indian States/UTs mapping for canonical title normalization
INDIAN_STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
    "Chandigarh", "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka",
    "Kerala", "Ladakh", "Lakshadweep", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
]

class FormLA7Extractor:
    """
    Precision extraction engine for the standardized 2-page Form LA-7 PDF template
    ('Land Acquisition Project Data Sheet').
    
    Supports both standard Form LA-7 layout variants:
    1. Inline Layout: Labels on the left, box-grids on the right (same row).
    2. Stacked Layout: Labels on top, box-grids on the next row below.
    """
    
    # --- Layout A: Inline Layout Coordinates ---
    BOX_REGIONS_INLINE = {
        'project_id': (1, 195.0, 220.0, 160.0, 545.0),
        'state': (1, 288.0, 312.0, 160.0, 470.0),
        'district': (1, 318.0, 342.0, 160.0, 470.0),
        'latitude': (1, 458.0, 480.0, 140.0, 275.0),
        'longitude': (1, 458.0, 480.0, 370.0, 505.0),
        'cost': (2, 168.0, 192.0, 195.0, 390.0),
        'land_area': (2, 198.0, 222.0, 195.0, 390.0),
        'fund_pct': (2, 408.0, 430.0, 195.0, 315.0),
        'sec11_days': (2, 490.0, 512.0, 240.0, 340.0),
        'comp_mult': (2, 520.0, 545.0, 240.0, 340.0),
        'affected_families': (2, 550.0, 575.0, 240.0, 360.0),
        'dispute_rate': (2, 582.0, 606.0, 240.0, 360.0),
    }

    CHECKBOX_GROUPS_INLINE = {
        'project_type': {
            'Highway': (1, 164.4, 239.2),
            'Railway': (1, 255.1, 239.2),
            'Irrigation': (1, 345.8, 239.2),
            'Power': (1, 164.4, 269.6),
            'Urban Dev.': (1, 255.1, 269.6),
            'Other': (1, 345.8, 269.6),
        },
        'terrain_type': {
            'Plain': (1, 164.4, 362.3),
            'Rural / Agricultural': (1, 272.1, 362.3),
            'Hilly': (1, 379.8, 362.3),
            'Forest / Tribal': (1, 164.4, 392.6),
            'Urban / Congested': (1, 272.1, 392.6),
            'Coastal': (1, 379.8, 392.6),
        },
        'road_type': {
            'Auto-Detect (PMGSY)': (1, 164.4, 501.4),
            'PMGSY Connected': (1, 300.5, 501.4),
            'State Highway': (1, 164.4, 531.8),
            'Not Connected': (1, 300.5, 531.8),
        },
        'sia_approval_status': {
            'Approved': (2, 198.4, 293.7),
            'Pending': (2, 283.5, 293.7),
            'Rejected': (2, 198.4, 324.0),
            'Not Applicable': (2, 283.5, 324.0),
        },
        'forest_clearance_status': {
            'Not Required': (2, 198.4, 354.3),
            'Pending': (2, 283.5, 354.3),
            'Approved': (2, 198.4, 384.7),
            'Rejected': (2, 283.5, 384.7),
        },
        'local_protest_flag': {
            'Yes': (2, 243.8, 626.7),
            'No': (2, 294.8, 626.7),
        }
    }
    SITE_ADDRESS_REGION_INLINE = (1, 555.0, 573.0, 180.0, 545.0)

    # --- Layout B: Stacked Layout Coordinates ---
    BOX_REGIONS_STACKED = {
        'project_id': (1, 162.0, 185.0, 50.0, 450.0),
        'state': (1, 250.0, 272.0, 50.0, 450.0),
        'district': (1, 280.0, 305.0, 50.0, 450.0),
        'latitude': (1, 395.0, 418.0, 50.0, 200.0),
        'longitude': (1, 395.0, 418.0, 250.0, 450.0),
        'cost': (2, 162.0, 185.0, 50.0, 250.0),
        'land_area': (2, 192.0, 215.0, 50.0, 250.0),
        'fund_pct': (2, 368.0, 390.0, 50.0, 200.0),
        'sec11_days': (2, 425.0, 448.0, 50.0, 150.0),
        'comp_mult': (2, 456.0, 478.0, 50.0, 150.0),
        'affected_families': (2, 487.0, 510.0, 50.0, 150.0),
        'dispute_rate': (2, 518.0, 540.0, 50.0, 150.0),
    }

    CHECKBOX_GROUPS_STACKED = {
        'project_type': {
            'Highway': (1, 51.0, 202.4),
            'Railway': (1, 192.8, 202.4),
            'Irrigation': (1, 334.5, 202.4),
            'Power': (1, 51.0, 218.3),
            'Urban Dev.': (1, 192.8, 218.3),
            'Other': (1, 334.5, 218.3),
        },
        'terrain_type': {
            'Plain': (1, 51.0, 321.4),
            'Rural / Agricultural': (1, 192.8, 321.4),
            'Hilly': (1, 334.5, 321.4),
            'Forest / Tribal': (1, 51.0, 337.3),
            'Urban / Congested': (1, 192.8, 337.3),
            'Coastal': (1, 334.5, 337.3),
        },
        'road_type': {
            'Auto-Detect (PMGSY)': (1, 51.0, 436.5),
            'PMGSY Connected': (1, 221.1, 436.5),
            'State Highway': (1, 51.0, 452.4),
            'Not Connected': (1, 221.1, 452.4),
        },
        'sia_approval_status': {
            'Approved': (2, 51.0, 261.6),
            'Pending': (2, 221.1, 261.6),
            'Rejected': (2, 51.0, 277.5),
            'Not Applicable': (2, 221.1, 277.5),
        },
        'forest_clearance_status': {
            'Not Required': (2, 51.0, 318.9),
            'Pending': (2, 221.1, 318.9),
            'Approved': (2, 51.0, 334.8),
            'Rejected': (2, 221.1, 334.8),
        },
        'local_protest_flag': {
            'Yes': (2, 51.0, 559.0),
            'No': (2, 164.4, 559.0),
        }
    }
    SITE_ADDRESS_REGION_STACKED = (1, 490.0, 512.0, 50.0, 545.0)

    @classmethod
    def _normalize_district(cls, state: Optional[str], raw_district: Optional[str]) -> Optional[str]:
        if not raw_district:
            return None
        raw_clean = re.sub(r'[\s\-_]', '', raw_district.lower())
        if not raw_clean:
            return None
        
        # Check district reference mapping in district_coordinates.json
        if os.path.exists("district_coordinates.json"):
            try:
                with open("district_coordinates.json", "r", encoding="utf-8") as f:
                    d_coords = json.load(f)
                
                # Check under matching state first
                if state:
                    st_norm = state.strip().lower()
                    for k in d_coords.keys():
                        if "|" in k:
                            st, dt = k.split("|", 1)
                            if st.lower().strip() == st_norm:
                                if re.sub(r'[\s\-_]', '', dt.lower()) == raw_clean:
                                    return dt.strip()
                
                # Global check across all districts
                for k in d_coords.keys():
                    if "|" in k:
                        _, dt = k.split("|", 1)
                        if re.sub(r'[\s\-_]', '', dt.lower()) == raw_clean:
                            return dt.strip()
            except Exception:
                pass

        return raw_district.title()

    @classmethod
    def extract_from_bytes(cls, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts all project parameters from PDF bytes with auto-layout detection.
        Raises ValueError with descriptive message if document is invalid.
        """
        try:
            stream = io.BytesIO(pdf_bytes)
            with pdfplumber.open(stream) as pdf:
                # 1. Validation: Must have exactly 2 pages
                if len(pdf.pages) != 2:
                    raise ValueError(
                        f"Invalid Form LA-7 document: Expected exactly 2 pages, found {len(pdf.pages)} pages."
                    )
                
                # 2. Validation: Header text markers on both pages
                p1_text = pdf.pages[0].extract_text() or ""
                p2_text = pdf.pages[1].extract_text() or ""
                
                if "FORM LA-7" not in p1_text or "PROJECT DATA INTAKE SHEET" not in p1_text:
                    raise ValueError(
                        "Uploaded PDF does not match Form LA-7 template header (Page 1 missing 'FORM LA-7' or 'PROJECT DATA INTAKE SHEET')."
                    )
                if "FORM LA-7" not in p2_text or "PROJECT DATA INTAKE SHEET" not in p2_text:
                    raise ValueError(
                        "Uploaded PDF does not match Form LA-7 template header (Page 2 missing 'FORM LA-7' or 'PROJECT DATA INTAKE SHEET')."
                    )
                
                p1 = pdf.pages[0]
                p2 = pdf.pages[1]
                pages = [p1, p2]
                
                # 3. Detect Layout Mode (Stacked vs Inline)
                p1_words = p1.extract_words()
                id_word = next((w for w in p1_words if "Identifier" in w["text"]), None)
                if id_word and id_word["top"] < 180.0:
                    box_regions = cls.BOX_REGIONS_STACKED
                    checkbox_groups = cls.CHECKBOX_GROUPS_STACKED
                    site_address_region = cls.SITE_ADDRESS_REGION_STACKED
                    layout_mode = "stacked"
                else:
                    box_regions = cls.BOX_REGIONS_INLINE
                    checkbox_groups = cls.CHECKBOX_GROUPS_INLINE
                    site_address_region = cls.SITE_ADDRESS_REGION_INLINE
                    layout_mode = "inline"

                raw_extracted: Dict[str, Any] = {}
                failed_fields: List[str] = []

                # --- 4. Extract Box Grid Fields ---
                for field_name, (page_idx, y_min, y_max, x_min, x_max) in box_regions.items():
                    page = pages[page_idx - 1]
                    # Filter characters strictly within the bounding region
                    chars = [
                        c for c in page.chars 
                        if y_min <= c['top'] <= y_max and x_min <= c['x0'] <= x_max
                    ]
                    chars = sorted(chars, key=lambda c: c['x0'])
                    val_str = "".join(c['text'] for c in chars).strip()
                    raw_extracted[field_name] = val_str if val_str else None

                # --- 5. Extract Checkbox Groups ---
                for group_name, options in checkbox_groups.items():
                    marked_options = []
                    for opt_label, (page_idx, x_cb, y_cb) in options.items():
                        page = pages[page_idx - 1]
                        # Check inside or around checkbox for 'X', checkmark, or non-whitespace tick
                        marks = [
                            c for c in page.chars
                            if abs(c['top'] - y_cb) < 8.0 and abs(c['x0'] - x_cb) < 9.0
                            and c['text'].strip().upper() in ['X', 'V', '✓', '✔']
                        ]
                        if marks:
                            marked_options.append(opt_label)
                    
                    if len(marked_options) == 1:
                        raw_extracted[group_name] = marked_options[0]
                    else:
                        raw_extracted[group_name] = None

                # --- 6. Extract Site Address Text ---
                p1_addr_words = [
                    w for w in p1.extract_words()
                    if site_address_region[1] <= w['top'] <= site_address_region[2]
                    and w['x0'] >= site_address_region[3]
                ]
                p1_addr_words = sorted(p1_addr_words, key=lambda w: w['x0'])
                raw_extracted['site_address'] = " ".join(w['text'] for w in p1_addr_words).strip() or None

                # --- 7. Clean, Normalize, and Strictly Validate Values ---
                cleaned_fields: Dict[str, Any] = {}

                # A. Project Identifier
                proj_id = raw_extracted.get('project_id')
                cleaned_fields['inp-project-id'] = proj_id if proj_id else None

                # B. Sector / Type
                p_type = raw_extracted.get('project_type')
                if p_type:
                    type_mapping = {
                        'Highway': 'Highway',
                        'Railway': 'Railway',
                        'Power': 'Renewable_Energy',
                        'Irrigation': 'Irrigation',
                        'Urban Dev.': 'Metro',
                        'Other': 'Highway'
                    }
                    cleaned_fields['inp-project-type'] = type_mapping.get(p_type, p_type)
                else:
                    cleaned_fields['inp-project-type'] = None

                # C. State / UT (normalize case and match to standard states)
                state_raw = raw_extracted.get('state')
                matched_state = None
                if state_raw:
                    norm_state = re.sub(r'[\s\-_]', '', state_raw.lower().replace('&', 'and'))
                    for st in INDIAN_STATES:
                        st_clean = re.sub(r'[\s\-_]', '', st.lower().replace('&', 'and'))
                        if st_clean == norm_state or norm_state in st_clean:
                            matched_state = st
                            break
                    cleaned_fields['inp-state'] = matched_state or state_raw.title()
                else:
                    cleaned_fields['inp-state'] = None

                # D. District (Normalized against standard district registry)
                dist_raw = raw_extracted.get('district')
                cleaned_fields['inp-district'] = cls._normalize_district(matched_state or cleaned_fields.get('inp-state'), dist_raw)

                # E. Terrain Type
                terrain_raw = raw_extracted.get('terrain_type')
                if terrain_raw:
                    terrain_mapping = {
                        'Rural / Agricultural': 'Rural_Agri',
                        'Plain': 'Rural_Agri',
                        'Forest / Tribal': 'Forest_Eco_Sensitive',
                        'Hilly': 'Hilly',
                        'Urban / Congested': 'Urban',
                        'Coastal': 'Rural_Agri'
                    }
                    cleaned_fields['inp-terrain'] = terrain_mapping.get(terrain_raw, 'Rural_Agri')
                else:
                    cleaned_fields['inp-terrain'] = None

                # F. Latitude (numeric validation)
                lat_raw = raw_extracted.get('latitude')
                if lat_raw:
                    try:
                        lat_val = float(lat_raw)
                        if 6.0 <= lat_val <= 38.0:
                            cleaned_fields['inp-latitude'] = round(lat_val, 4)
                        else:
                            failed_fields.append('latitude')
                            cleaned_fields['inp-latitude'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('latitude')
                        cleaned_fields['inp-latitude'] = None
                else:
                    cleaned_fields['inp-latitude'] = None

                # G. Longitude (numeric validation)
                lon_raw = raw_extracted.get('longitude')
                if lon_raw:
                    try:
                        lon_val = float(lon_raw)
                        if 68.0 <= lon_val <= 98.0:
                            cleaned_fields['inp-longitude'] = round(lon_val, 4)
                        else:
                            failed_fields.append('longitude')
                            cleaned_fields['inp-longitude'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('longitude')
                        cleaned_fields['inp-longitude'] = None
                else:
                    cleaned_fields['inp-longitude'] = None

                # H. Road Connectivity
                road_raw = raw_extracted.get('road_type')
                if road_raw:
                    road_mapping = {
                        'Auto-Detect (PMGSY)': '',
                        'PMGSY Connected': 'PMGSY Connected',
                        'State Highway': 'State Highway',
                        'Not Connected': 'No Formal Road'
                    }
                    cleaned_fields['inp-road-type'] = road_mapping.get(road_raw, road_raw)
                else:
                    cleaned_fields['inp-road-type'] = None

                # I. Site Address
                cleaned_fields['inp-site-address'] = raw_extracted.get('site_address')

                # J. Capital Outlay (INR Crore)
                cost_raw = raw_extracted.get('cost')
                if cost_raw:
                    try:
                        cost_val = float(cost_raw)
                        if cost_val > 0:
                            cleaned_fields['inp-cost'] = round(cost_val, 2)
                        else:
                            failed_fields.append('cost')
                            cleaned_fields['inp-cost'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('cost')
                        cleaned_fields['inp-cost'] = None
                else:
                    cleaned_fields['inp-cost'] = None

                # K. Land Extent (Hectares)
                land_raw = raw_extracted.get('land_area')
                if land_raw:
                    try:
                        land_val = float(land_raw)
                        if land_val > 0:
                            cleaned_fields['inp-land-area'] = round(land_val, 2)
                        else:
                            failed_fields.append('land_area')
                            cleaned_fields['inp-land-area'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('land_area')
                        cleaned_fields['inp-land-area'] = None
                else:
                    cleaned_fields['inp-land-area'] = None

                # L. SIA Approval Status
                sia_raw = raw_extracted.get('sia_approval_status')
                if sia_raw:
                    sia_mapping = {
                        'Approved': 'Approved',
                        'Pending': 'Pending',
                        'Rejected': 'Rejected',
                        'Not Applicable': 'Exempted'
                    }
                    cleaned_fields['inp-sia-status'] = sia_mapping.get(sia_raw, sia_raw)
                else:
                    cleaned_fields['inp-sia-status'] = None

                # M. Forest Clearance Status
                forest_raw = raw_extracted.get('forest_clearance_status')
                if forest_raw:
                    forest_mapping = {
                        'Not Required': 'Not_Required',
                        'Pending': 'Pending',
                        'Approved': 'Approved',
                        'Rejected': 'Rejected'
                    }
                    cleaned_fields['inp-forest-status'] = forest_mapping.get(forest_raw, forest_raw)
                else:
                    cleaned_fields['inp-forest-status'] = None

                # N. Fund Disbursed (%)
                fund_raw = raw_extracted.get('fund_pct')
                if fund_raw:
                    try:
                        fund_val = float(fund_raw)
                        if 0.0 <= fund_val <= 100.0:
                            cleaned_fields['inp-fund-pct'] = round(fund_val, 2)
                        else:
                            failed_fields.append('fund_pct')
                            cleaned_fields['inp-fund-pct'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('fund_pct')
                        cleaned_fields['inp-fund-pct'] = None
                else:
                    cleaned_fields['inp-fund-pct'] = None

                # O. Sec. 11 Days Elapsed
                sec11_raw = raw_extracted.get('sec11_days')
                if sec11_raw:
                    try:
                        sec11_val = int(round(float(sec11_raw)))
                        if sec11_val >= 0:
                            cleaned_fields['inp-sec11-days'] = sec11_val
                        else:
                            failed_fields.append('sec11_days')
                            cleaned_fields['inp-sec11-days'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('sec11_days')
                        cleaned_fields['inp-sec11-days'] = None
                else:
                    cleaned_fields['inp-sec11-days'] = None

                # P. Sec. 26 Multiplier Demand
                mult_raw = raw_extracted.get('comp_mult')
                if mult_raw:
                    try:
                        mult_val = float(mult_raw)
                        if mult_val >= 1.0:
                            cleaned_fields['inp-comp-mult'] = round(mult_val, 2)
                        else:
                            failed_fields.append('comp_mult')
                            cleaned_fields['inp-comp-mult'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('comp_mult')
                        cleaned_fields['inp-comp-mult'] = None
                else:
                    cleaned_fields['inp-comp-mult'] = None

                # Q. PAFs Count
                paf_raw = raw_extracted.get('affected_families')
                if paf_raw:
                    try:
                        paf_val = int(round(float(paf_raw)))
                        if paf_val >= 0:
                            cleaned_fields['inp-affected-families'] = paf_val
                        else:
                            failed_fields.append('affected_families')
                            cleaned_fields['inp-affected-families'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('affected_families')
                        cleaned_fields['inp-affected-families'] = None
                else:
                    cleaned_fields['inp-affected-families'] = None

                # R. Sec. 15 Title Dispute Rate (%)
                disp_raw = raw_extracted.get('dispute_rate')
                if disp_raw:
                    try:
                        disp_val = float(disp_raw)
                        if 0.0 <= disp_val <= 100.0:
                            cleaned_fields['inp-dispute-rate'] = round(disp_val, 2)
                        else:
                            failed_fields.append('dispute_rate')
                            cleaned_fields['inp-dispute-rate'] = None
                    except (ValueError, TypeError):
                        failed_fields.append('dispute_rate')
                        cleaned_fields['inp-dispute-rate'] = None
                else:
                    cleaned_fields['inp-dispute-rate'] = None

                # S. Local Agitation / Protest Flag
                protest_raw = raw_extracted.get('local_protest_flag')
                if protest_raw == 'Yes':
                    cleaned_fields['inp-protest-flag'] = True
                elif protest_raw == 'No':
                    cleaned_fields['inp-protest-flag'] = False
                else:
                    cleaned_fields['inp-protest-flag'] = None

                # --- 8. Validation: Must have at least 1 extractable field ---
                filled_count = sum(1 for v in cleaned_fields.values() if v is not None)
                total_fields = len(cleaned_fields)
                
                if filled_count == 0:
                    raise ValueError(
                        "No valid project data fields could be extracted from this Form LA-7 document. The form appears entirely blank."
                    )

                return {
                    "status": "success",
                    "template_name": "Form LA-7 (RFCTLARR Act 2013)",
                    "layout_mode": layout_mode,
                    "filled_count": filled_count,
                    "total_fields": total_fields,
                    "fields": cleaned_fields,
                    "raw_extracted": raw_extracted,
                    "failed_fields": failed_fields
                }

        except ValueError as ve:
            raise ve
        except Exception as e:
            logger.error(f"Failed to extract Form LA-7 PDF: {e}", exc_info=True)
            raise ValueError(f"Could not parse PDF document: {e}")
