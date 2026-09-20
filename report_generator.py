"""
AthleteGuard AI — Clinical PDF Report Generator.
Generates an executive, single-page clinical biomechanical screening report
for coaches, athletic trainers, and sports physiotherapists using ReportLab.
"""

import io
import os
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(athlete_id: str,
                        sport: str,
                        video_name: str,
                        result: Dict[str, Any],
                        output_path: Optional[str] = None) -> bytes:
    """
    Generate a concise, professional 1-page PDF screening report.
    
    Args:
        athlete_id: Athlete name or identifier
        sport: Sport / discipline (e.g. Long Jump, Volleyball)
        video_name: Source video filename
        result: Analysis result dictionary from process_video()
        output_path: Optional filepath to save the PDF
        
    Returns:
        PDF bytes buffer
    """
    buffer = io.BytesIO()
    target_dest = output_path if output_path else buffer
    
    # 1-page tight geometry: Letter size (612 x 792 pt), 28 pt margins
    doc = SimpleDocTemplate(
        target_dest,
        pagesize=letter,
        leftMargin=28,
        rightMargin=28,
        topMargin=24,
        bottomMargin=24
    )
    
    content_width = 556  # 612 - 56
    
    # Palette definition
    c_navy = colors.HexColor("#0F172A")
    c_blue = colors.HexColor("#1D4ED8")
    c_blue_light = colors.HexColor("#EFF6FF")
    c_gray_dark = colors.HexColor("#1E293B")
    c_gray_muted = colors.HexColor("#64748B")
    c_gray_bg = colors.HexColor("#F8FAFC")
    c_border = colors.HexColor("#CBD5E1")
    c_border_light = colors.HexColor("#E2E8F0")
    
    c_green = colors.HexColor("#059669")
    c_amber = colors.HexColor("#D97706")
    c_red = colors.HexColor("#DC2626")
    
    # Extract results
    overall_score = result.get('overall_risk_score', 0.0)
    overall_level = result.get('overall_risk_level', 'UNKNOWN')
    peak_risk = result.get('peak_risk', {})
    peak_score = peak_risk.get('score', 0.0)
    peak_frame = peak_risk.get('frame', 0)
    peak_phase = peak_risk.get('phase', 'UNKNOWN')
    
    landing_elevated = result.get('landing_risk_elevated', False)
    landing_alert = result.get('landing_alert', 'Normal')
    phase_data = result.get('phase_analysis', {})
    landing_avg = phase_data.get('LANDING', {}).get('average_risk_score', 0.0)
    
    drivers_info = result.get('risk_drivers', {})
    driver_stmt = drivers_info.get('statement', 'Kinematics evaluated within expected parameters.')
    primary_drivers = drivers_info.get('primary_drivers', [])
    safe_metrics = drivers_info.get('safe_metrics', [])
    
    last_res = result.get('risk_result', {})
    components = last_res.get('components', {})
    raw_vals = last_res.get('raw_values', {})
    
    # Risk color determination
    if overall_score <= 30.0:
        badge_color = c_green
        badge_bg = colors.HexColor("#ECFDF5")
    elif overall_score <= 60.0:
        badge_color = c_amber
        badge_bg = colors.HexColor("#FFFBEB")
    else:
        badge_color = c_red
        badge_bg = colors.HexColor("#FEF2F2")

    # Typography styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=c_navy
    )
    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_blue
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=c_gray_dark
    )
    sec_hdr_style = ParagraphStyle(
        'SecHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=c_navy
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=c_gray_dark
    )
    driver_style = ParagraphStyle(
        'DriverStmt',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#991B1B") if overall_score > 30 else colors.HexColor("#065F46")
    )
    tbl_hdr_style = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )
    tbl_cell_bold = ParagraphStyle(
        'TblCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=c_gray_dark
    )
    tbl_cell_body = ParagraphStyle(
        'TblCellBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=8.5,
        textColor=c_gray_dark
    )
    
    elements = []
    
    # -------------------------------------------------------------
    # 1. HEADER & METADATA BANNER
    # -------------------------------------------------------------
    header_left = [
        Paragraph("ATHLETEGUARD AI | Clinical Kinematic Screening", title_style),
        Paragraph("Landing Error Scoring System (LESS) & ACL Injury Risk Protocol", subtitle_style)
    ]
    
    now_str = datetime.now().strftime("%d %b %Y, %H:%M")
    meta_lines = f"""
    <b>Athlete:</b> {athlete_id} &nbsp;|&nbsp; <b>Sport:</b> {sport}<br/>
    <b>Date:</b> {now_str} &nbsp;|&nbsp; <b>Footage:</b> {video_name}
    """
    header_right = Paragraph(meta_lines, meta_style)
    
    header_table = Table(
        [[header_left, header_right]],
        colWidths=[316, 240]
    )
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=4, spaceAfter=6))
    
    # Extract extra kinematic parameters
    persp_raw = result.get('perspective', 'SAGITTAL')
    if isinstance(persp_raw, dict):
        persp_view = persp_raw.get('view_label', 'Sagittal View')
    else:
        persp_view = f"{str(persp_raw).title()} View"
    bench_info = result.get('benchmarks', {})
    bench_p = bench_info.get('summary', '')
    foot_strike = result.get('landing_foot_strike', 'Forefoot / Midfoot')

    # -------------------------------------------------------------
    # 2. EXECUTIVE RISK KPI & RISK DRIVER ATTRIBUTION BOX
    # -------------------------------------------------------------
    kpi_html = f"""
    <font size=7 color="#64748B">OVERALL MOVEMENT RISK</font><br/>
    <font size=18 color="{badge_color.hexval()}"><b>{overall_score:.1f}</b></font>
    <font size=9 color="#64748B">/ 100</font> &nbsp;
    <b><font size=8.5 color="{badge_color.hexval()}">[{overall_level}]</font></b><br/>
    <font size=7 color="#1E293B">
    <b>Peak:</b> {peak_score:.1f} (F:{peak_frame}, {peak_phase})<br/>
    <b>Landing:</b> {landing_avg:.1f} &nbsp;[{'ELEVATED' if landing_elevated else 'NORMAL'}]<br/>
    <b>Camera:</b> {persp_view}<br/>
    <b>Strike:</b> {foot_strike}
    </font>
    """
    kpi_p = Paragraph(kpi_html, body_style)
    
    # Right Column: Risk Drivers Narrative & Safe Mechanics
    driver_lines = [
        Paragraph("<b>Kinematic Risk Driver Attribution:</b>", sec_hdr_style),
        Paragraph(f"{driver_stmt}", driver_style)
    ]
    
    driver_bullets = []
    if bench_p:
        driver_bullets.append(f"<b>Normative Benchmarking:</b> {bench_p}")
    if primary_drivers:
        driver_bullets.append("<b>Elevating Factors:</b> " + ", ".join([f"{d['label']} ({d['raw_value']}°)" if d.get('raw_value') and 'symmetry' not in d.get('metric','') else f"{d['label']}" for d in primary_drivers]))
    if safe_metrics:
        driver_bullets.append("<b>Protective Factors:</b> " + ", ".join([d['label'] for d in safe_metrics[:3]]))
    
    if driver_bullets:
        driver_lines.append(Paragraph("<br/>".join(driver_bullets), body_style))
        
    exec_table = Table(
        [[kpi_p, driver_lines]],
        colWidths=[165, 391]
    )
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), badge_bg),
        ('BACKGROUND', (1,0), (1,0), c_gray_bg),
        ('BOX', (0,0), (0,0), 1, badge_color),
        ('BOX', (1,0), (1,0), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(exec_table)
    elements.append(Spacer(1, 5))
    
    # -------------------------------------------------------------
    # 3. PHASE-AWARE KINEMATIC PROGRESSION TABLE
    # -------------------------------------------------------------
    elements.append(Paragraph("Phase-Aware Kinematic Risk Progression", sec_hdr_style))
    elements.append(Spacer(1, 2))
    
    phase_table_data = [
        [
            Paragraph("Movement Phase", tbl_hdr_style),
            Paragraph("Frames", tbl_hdr_style),
            Paragraph("Avg Risk", tbl_hdr_style),
            Paragraph("Peak Risk", tbl_hdr_style),
            Paragraph("Risk Level", tbl_hdr_style),
            Paragraph("Key Biomechanical Assessment", tbl_hdr_style)
        ]
    ]
    
    phase_order = ["APPROACH", "TAKE_OFF", "FLIGHT", "LANDING"]
    phase_notes = {
        "APPROACH": "Horizontal runway posture, linear acceleration stability.",
        "TAKE_OFF": "Penultimate step knee flexion, plant angle, vertical impulse.",
        "FLIGHT": "Mid-air trunk posture, preparation for initial contact.",
        "LANDING": "Ground contact deceleration, knee valgus & hip absorption."
    }
    
    for p_name in phase_order:
        p_stats = phase_data.get(p_name, {})
        avg = p_stats.get('average_risk_score')
        pk = p_stats.get('peak_risk_score')
        f_count = p_stats.get('frame_count', 0)
        lvl = p_stats.get('risk_level', 'N/A')
        
        avg_str = f"{avg:.1f}/100" if avg is not None else "--"
        pk_str = f"{pk:.1f}" if pk is not None else "--"
        note = phase_notes.get(p_name, "")
        
        if p_name == "LANDING" and p_stats.get('risk_drivers', {}).get('statement'):
            note = p_stats['risk_drivers']['statement']
            
        phase_table_data.append([
            Paragraph(f"<b>{p_name.replace('_', ' ')}</b>", tbl_cell_bold),
            Paragraph(str(f_count), tbl_cell_body),
            Paragraph(avg_str, tbl_cell_body),
            Paragraph(pk_str, tbl_cell_body),
            Paragraph(f"<b>{lvl}</b>", tbl_cell_body),
            Paragraph(note, tbl_cell_body)
        ])
        
    phase_table = Table(
        phase_table_data,
        colWidths=[80, 42, 54, 52, 60, 268]
    )
    phase_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_navy),
        ('ALIGN', (1,0), (4,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border_light),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(phase_table)
    elements.append(Spacer(1, 5))
    
    # -------------------------------------------------------------
    # 4. LESS BIOMECHANICAL ITEMS MAPPING TABLE
    # -------------------------------------------------------------
    elements.append(Paragraph("Landing Error Scoring System (LESS) Items & Kinematics", sec_hdr_style))
    elements.append(Spacer(1, 2))
    
    less_table_data = [
        [
            Paragraph("LESS Biomechanical Item", tbl_hdr_style),
            Paragraph("Measured Value", tbl_hdr_style),
            Paragraph("Score", tbl_hdr_style),
            Paragraph("Status", tbl_hdr_style),
            Paragraph("Clinical Evidence & Injury Criteria (Padua et al., 2009)", tbl_hdr_style)
        ]
    ]
    
    items_def = [
        ('knee_valgus', 'Frontal Knee Valgus (FPPA)', 'knee_valgus', '°', 'Medial knee collapse inside ankle axis is the primary ACL injury mechanism.'),
        ('knee_angle', 'Knee Flexion at Landing', 'knee_angle', '°', 'Landing flexion > 45° reduces ground reaction forces; stiff landing elevates risk.'),
        ('hip_angle', 'Hip Flexion Attenuation', 'hip_angle', '°', 'Adequate hip flexion engages posterior chain; stiff landing shifts shock to knee.'),
        ('trunk_angle', 'Trunk Flexion Posture', 'trunk_angle', '°', 'Moderate forward lean aligns center of mass; upright posture shocks spine.'),
        ('ankle_dorsiflexion', 'Ankle Dorsiflexion (LESS 5)', 'ankle_dorsiflexion', '°', 'Elastic dorsiflexion (70°-105°) absorbs ground impact; rigid contact jolts knee.'),
        ('symmetry_knee', 'Bilateral Knee Symmetry', 'symmetry_knee', '%', 'Asymmetrical limb impact overloads dominant or early-landing joint structure.'),
        ('symmetry_hip', 'Pelvic / Hip Symmetry', 'symmetry_hip', '%', 'Pelvic drop or lateral tilt reflects abductor / gluteus medius insufficiency.'),
        ('temporal_stability', 'Movement Smoothness', 'temporal_stability', '%', 'Smooth, continuous joint deceleration without high-frequency kinematic jitter.')
    ]
    
    for metric_key, item_name, raw_key, unit, clinical_rule in items_def:
        sc = components.get(metric_key, 0.0)
        raw = raw_vals.get(raw_key)
        raw_display = f"{raw:.1f}{unit}" if raw is not None else "--"
        
        if sc >= 75.0:
            st_text = "OPTIMAL"
            st_color = c_green
        elif sc >= 50.0:
            st_text = "MODERATE"
            st_color = c_amber
        else:
            st_text = "DEFICIT"
            st_color = c_red
            
        less_table_data.append([
            Paragraph(f"<b>{item_name}</b>", tbl_cell_bold),
            Paragraph(raw_display, tbl_cell_body),
            Paragraph(f"{sc:.0f}/100", tbl_cell_body),
            Paragraph(f"<b><font color='{st_color.hexval()}'>{st_text}</font></b>", tbl_cell_body),
            Paragraph(clinical_rule, tbl_cell_body)
        ])
        
    less_table = Table(
        less_table_data,
        colWidths=[140, 68, 48, 56, 244]
    )
    less_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_blue),
        ('ALIGN', (1,0), (3,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, c_border_light),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(less_table)
    elements.append(Spacer(1, 6))
    
    # -------------------------------------------------------------
    # 5. CLINICAL & COACHING CORRECTIVE DRILLS
    # -------------------------------------------------------------
    elements.append(Paragraph("Physiotherapist & Coaching Corrective Action Protocols", sec_hdr_style))
    elements.append(Spacer(1, 2))
    
    drills = []
    if any(d.get('metric', '') == 'knee_valgus' for d in primary_drivers) or components.get('knee_valgus', 100) < 60:
        drills.append("<b>1. Valgus Neuromuscular Retraining:</b> Mini-band lateral walks, banded drop-jumps with mirror feedback to prevent medial collapse.")
    if any(d.get('metric', '') == 'knee_angle' for d in primary_drivers) or components.get('knee_angle', 100) < 60:
        drills.append("<b>2. Soft Landing Mechanics:</b> Box drop-and-stick drills emphasizing knee flexion > 60° with silent, compliant foot strikes.")
    if any(d.get('metric', '') in ['symmetry_knee', 'symmetry_hip'] for d in primary_drivers) or components.get('symmetry_hip', 100) < 60:
        drills.append("<b>3. Bilateral Symmetry & Gluteus Medius Strength:</b> Single-leg Romanian deadlifts, side planks, and unilateral deceleration balance drills.")
    if not drills:
        drills.append("<b>1. Maintenance Protocol:</b> Continue plyometric volume progression while maintaining neutral knee alignment.")
        drills.append("<b>2. Velocity Loading:</b> Progressively introduce competitive approach speeds while monitoring landing compliance.")
        
    drills_text = "<br/>".join(drills[:2])
    rec_table = Table(
        [[Paragraph(drills_text, body_style)]],
        colWidths=[content_width]
    )
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), c_blue_light),
        ('BOX', (0,0), (0,0), 1, c_blue),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 6))
    
    # -------------------------------------------------------------
    # 6. FOOTER / DISCLAIMER
    # -------------------------------------------------------------
    footer_text = (
        "CONFIDENTIAL & ATHLETIC PERFORMANCE SCREENING ONLY. AthleteGuard AI evaluates joint kinematics and "
        "injury risk indicators for athletic training optimization. Not intended as a substitute for clinical medical diagnosis or treatment."
    )
    elements.append(Paragraph(f"<font size=6 color='#64748B'>{footer_text}</font>", body_style))
    
    # Build Document
    doc.build(elements)
    
    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    else:
        buffer.seek(0)
        return buffer.getvalue()
