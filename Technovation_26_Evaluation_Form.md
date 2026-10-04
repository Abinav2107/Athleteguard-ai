| <img src="https://kiot.ac.in/wp-content/uploads/2021/04/logo.png" width="80" alt="KIOT Logo"/> | **KNOWLEDGE INSTITUTE OF TECHNOLOGY, SALEM**<br/>*(An Autonomous Institution)*<br/>**iStartKIOT MXincubator Foundation**<br/>**Technovation’26** | <img src="https://iic.mic.gov.in/assets/images/iic-logo.png" width="90" alt="IIC Logo"/> |
| :--- | :--- | :--- |

| **Date** | 04-10-2026 | **Day** | 1 | **Team No.** | *(Fill Team No.)* |
| **Team Name** | **AthleteGuard AI** |
| **Project Title** | **AthleteGuard AI — Clinical Movement Screening, Limb Symmetry Index (LSI), & Post-Op Orthopedic Surgical Clearance Decision Support System** |

---

### **Problem & User**

| **Question** | **Evaluation Response** |
| :--- | :--- |
| **Which sports health problem are you solving (injury, rehab, fatigue, nutrition, performance, mental wellbeing)?** | **Non-Contact Lower-Extremity Injuries & Post-Operative ACL Rehabilitation Clearance.**<br/>Specifically addressing acute non-contact Anterior Cruciate Ligament (ACL) ruptures, graft re-tears, patellofemoral syndrome, and compensatory asymmetric overloading during jump-landing and deceleration. |
| **What happens today without your solution, and why is that not good enough?** | **1. Subjective Visual Checklists:** Coaches and clinicians rely on the naked eye or paper LESS sheets, yielding $>30\%$ inter-rater variability and missing microsecond dynamic valgus collapses ($<50\text{ ms}$).<br/>**2. Prohibitive Lab Equipment Costs:** Gold-standard optical 3D motion capture labs (Vicon/Qualisys) and force plates cost \$50,000–\$150,000, require physical markers, and take hours of setup.<br/>**3. Arbitrary Return-to-Play (RTP) Timelines:** Orthopedic surgeons frequently clear post-ACL reconstruction patients based solely on elapsed calendar time (e.g. *"it's been 9 months"*), resulting in a devastating **20%–30% secondary graft re-tear rate**. Without objective bilateral Limb Symmetry Index (LSI) data, early quad avoidance and contralateral overload go undetected. |
| **How will you measure success (e.g., injury risk detected earlier, accuracy of form correction)?** | **1. Objective Limb Symmetry Index (LSI $\ge 90\%$):** Conformance with the *Grindem et al. (2016)* clinical benchmark, which proves that achieving $\ge 90\%$ LSI reduces secondary ACL re-rupture risk by **51% per month delayed**.<br/>**2. High-Speed Biomechanical Deficit Detection:** Automatic identification of dynamic frontal valgus collapse (FPPA $>10^\circ$) and landing stiffness ($<60^\circ$ knee flexion) in under **60 seconds**.<br/>**3. Goniometric Accuracy:** Angular kinematic precision within $\pm 3^\circ$ compared against laboratory manual goniometer benchmarks. |
| **Who is the primary user: athlete, coach, physiotherapist, or sports doctor?** | **Primary:** Orthopedic Surgeons, Sports Medicine Physicians, and Clinical Physical Therapists (surgical clearance, milestone tracking, and CPT-coded reimbursement).<br/>**Secondary:** Head Athletic Trainers, Strength & Conditioning Coaches, and Competitive Athletes. |
| **Flow Chart Available (Yes / No)** | **Yes** *(Included in Project Dossier & Visual Architecture)* |
| **Architecture Diagram (Yes / No)** | **Yes** *(Included in Project Dossier & Visual Architecture)* |

---

### **Data**

| **Question** | **Evaluation Response** |
| :--- | :--- |
| **What data do you need (wearables, video, EMG, heart rate, injury records, nutrition logs)?** | **Standard 2D RGB Video Footage (MP4/MOV/AVI, 30–60 FPS)** captured via any smartphone or tablet camera showing a full-body jump-landing or deceleration movement (e.g., Drop Vertical Jump, Long Jump, Volleyball Spike landing).<br/>*Zero wearable sensors, physical skin markers, or special rigs required.* |
| **Is this data publicly available or will you generate or simulate it? Who owns it?** | **1. Kinematic Ground Truth:** Calibrated against open academic benchmarks from the **Padua et al. (2009) LESS protocol** and IAAF biomechanical datasets.<br/>**2. Patient / Athlete Footage:** Recorded on-demand by the clinician or athlete. Patient data ownership strictly remains with the patient/hospital; video is processed transiently in local volatile memory. |
| **Is the dataset large and balanced enough, and does it cover different ages, genders, and sports?** | **1. Base Computer Vision Model:** Google MediaPipe BlazePose was trained on over **25,000+ diverse full-body motion images** encompassing diverse ages, BMIs, genders, skin tones, and camera perspectives.<br/>**2. Multi-Sport & Clinical Calibration:** Calibrated across **5 distinct sport/movement archetypes** (Clinical Drop Vertical Jump, Track Long Jump, Volleyball Spike/Block, Basketball Deceleration, and Soccer Cutting) with adaptive threshold normalizations. |

---

### **AI Approach**

| **Question** | **Evaluation Response** |
| :--- | :--- |
| **Which AI technique fits (classification, time-series prediction, computer vision/pose estimation, anomaly detection, LLM-based advice)?** | **Deep Learning 3D Computer Vision (MediaPipe 33 3D Pose Landmarks)** combined with:<br/>**1. Deterministic Biomechanical Kinematic Engine:** Vector dot-product joint angle calculation, Frontal Plane Projection Angle (FPPA) deviation, and vertical ankle velocity phase segmentation (Approach, Takeoff, Flight, Landing).<br/>**2. Clinical Decision Support Heuristics:** Weighted Limb Symmetry Index (LSI %) formulation and Padua LESS item attribution.<br/>**3. Generative Clinical AI (Google Gemini 2.5 Flash API):** Synthesizes measured kinematic deficits into periodized 4-week clinical corrective exercise protocols. |
| **What is your baseline, and how will you prove the AI adds value?** | **Baseline:** Manual observational assessment by a physiotherapist using a handheld goniometer and paper-based LESS checklist (15–20 min per patient, high subjectivity, blind to millisecond dynamic valgus spikes).<br/>**AI Value-Add:** AthleteGuard AI computes 33 landmark coordinates across all 4 movement phases, extracts bilateral symmetry (LSI %), flags micro-collapses in $<60$ seconds, and auto-generates a signed CPT-coded clinical clearance PDF. |
| **How will you validate the model (metrics, test split, expert feedback)?** | **1. Kinematic Precision Validation:** Benchmarking calculated joint angles against manual goniometric measurements across trial repetitions.<br/>**2. Phase Boundary Temporal Accuracy:** Validating Initial Contact (IC) and Peak Knee Flexion (PF) against manual frame-by-frame sports video scrubbers.<br/>**3. Expert Clinical Validation:** Review and feedback by orthopedic surgeons and sports physical therapists on clearance certificate outputs and CPT billing compliance (CPT 97750, 98975, 98977). |
| **What measurable outcome will prove that your solution works?** | **1. Real-time Inference Throughput:** $\ge 25\text{ FPS}$ on standard CPU/browser hardware.<br/>**2. $100\%$ Detection of High-Risk Valgus:** Complete capture of dynamic knee valgus collapses exceeding sport safety thresholds ($>10^\circ$).<br/>**3. Objective Bilateral Differentiation:** Clear statistical separation between healthy limbs ($\ge 90\%$ LSI) and post-op deficient limbs ($<80\%$ LSI). |

---

### **Safety & Ethics**

| **Question** | **Evaluation Response** |
| :--- | :--- |
| **How will you protect personal health data and get consent?** | **1. Localized / Edge Processing:** Video frames are analyzed transiently in volatile application memory and discarded. No raw footage is transmitted to third-party cloud storage.<br/>**2. Patient De-Identification:** Patient records use anonymized Medical Record Numbers (MRN) or Athlete IDs.<br/>**3. Full Patient Consent & Control:** Built-in *"Reset Database & Clear History"* utility allows instant deletion of all local screening logs. |
| **What are the risks of a wrong prediction, and how will you show uncertainty?** | **Risks:** Premature return-to-play clearance leading to graft rupture, or false-positive holds delaying an athlete's career.<br/>**Uncertainty Mitigation:**<br/>• Landmark visibility thresholding (coordinates with $<0.5$ confidence are rejected or flagged).<br/>• Automated camera perspective detection (Sagittal vs Frontal vs Oblique) to warn clinicians when frontal angles are estimated from lateral views.<br/>• Three-tier risk categorization (`LOW`, `MODERATE`, `HIGH`) paired with plain-language biomechanical attribution explaining the exact mechanical driver. |
| **How will you make clear this is decision support and not a medical diagnosis?** | **1. Prominent Ethical Disclaimers:** Burnt into every interface screen, report footer, and PDF header: *"AthleteGuard AI is a biomechanical movement screening and clinical decision support tool, not an autonomous medical diagnostic device. Final surgical clearance and return-to-play decisions remain under the professional judgment of the licensed orthopedic surgeon or physical therapist."*<br/>**2. Mandatory Physician Sign-Off Block:** The surgical clearance certificate requires the attending physician's manual physical signature, date, and Medical License / NPI verification before being entered into hospital EMRs or submitted for CPT reimbursement. |

---

| **Team Leader** | **Mentor** | **Event Coordinator** |
| :---: | :---: | :---: |
| *(Signature)* | *(Signature)* | *(Signature)* |
