"""
Seeds the database with entirely SYNTHETIC patients, notes, and accounts.
Run once: `python seed_data.py`

Default accounts (username / password):
  admin      / admin123      (role: admin)
  dr_raza    / doctor123     (role: doctor)
  patient1   / patient123    (role: patient, owns patient_id 1 - Ayesha Khan)
  patient2   / patient123    (role: patient, owns patient_id 2 - Bilal Ahmed)

Change these before exposing the app beyond localhost.
"""
from database import init_db, SessionLocal, User, Patient, ClinicalNote
from auth import hash_password

SYNTHETIC_PATIENTS = [
    {
        "mrn": "MRN-0001",
        "full_name": "Ayesha Khan",
        "dob": "1990-04-12",
        "sex": "F",
        "primary_diagnosis": "Type 2 Diabetes Mellitus",
        "medications": "Metformin 500mg BID, Atorvastatin 10mg OD",
        "allergies": "Penicillin",
        "notes": [
            ("Dr. Amina Raza", "progress_note",
             "Patient reports improved fasting glucose readings (110-130 mg/dL) since "
             "last visit. Continues Metformin 500mg twice daily. No hypoglycemic "
             "episodes reported. Advised to continue current diet plan and follow up "
             "in 3 months with repeat HbA1c."),
            ("Dr. Amina Raza", "lab_result",
             "HbA1c: 7.1% (down from 7.8% three months ago). LDL cholesterol 98 mg/dL, "
             "within target on current statin dose. Renal function normal."),
            ("Nurse Sana Malik", "nursing_note",
             "Patient asked about switching to a continuous glucose monitor. Provided "
             "educational materials. Patient mentioned mild anxiety about the diagnosis; "
             "referred to counseling resources on request."),
        ],
    },
    {
        "mrn": "MRN-0002",
        "full_name": "Bilal Ahmed",
        "dob": "1985-11-02",
        "sex": "M",
        "primary_diagnosis": "Hypertension, Generalized Anxiety Disorder",
        "medications": "Lisinopril 10mg OD, Sertraline 50mg OD",
        "allergies": "None known",
        "notes": [
            ("Dr. Faisal Iqbal", "progress_note",
             "Blood pressure well controlled at 128/82 on current Lisinopril dose. "
             "Patient reports anxiety symptoms have improved on Sertraline, sleep "
             "quality better. No side effects reported. Continue current regimen."),
            ("Dr. Faisal Iqbal", "progress_note",
             "Follow-up for GAD. PHQ-9 and GAD-7 scores both improved from baseline. "
             "Discussed continuing therapy alongside medication. Next review in 6 weeks."),
        ],
    },
]


def run():
    init_db()
    db = SessionLocal()
    try:
        if db.query(Patient).count() > 0:
            print("Database already seeded. Delete healthcare_demo.db to reseed.")
            return

        for idx, p in enumerate(SYNTHETIC_PATIENTS, start=1):
            patient = Patient(
                mrn=p["mrn"], full_name=p["full_name"], dob=p["dob"], sex=p["sex"],
                primary_diagnosis=p["primary_diagnosis"], medications=p["medications"],
                allergies=p["allergies"],
            )
            db.add(patient)
            db.flush()

            for author, note_type, content in p["notes"]:
                db.add(ClinicalNote(patient_id=patient.id, author=author, note_type=note_type, content=content))

            db.add(User(
                username=f"patient{idx}",
                hashed_password=hash_password("patient123"),
                role="patient",
                patient_id=patient.id,
            ))

        db.add(User(username="dr_raza", hashed_password=hash_password("doctor123"), role="doctor"))
        db.add(User(username="admin", hashed_password=hash_password("admin123"), role="admin"))

        db.commit()
        print("Seeded 2 synthetic patients, their notes, and demo accounts.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
