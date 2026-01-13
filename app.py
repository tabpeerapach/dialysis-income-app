# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# 1) FIXED POSITIONS (ตามชื่อ)
# -------------------------
RN_LEVEL_MAP = {
    "กาญจนา": "RN4",
    "ทัศนีย์": "RN4",
    "แก้วกาญจ์ตา": "RN3",
    "จุฑารัตน์": "RN2",
    "อัยลดา": "RN1",
    "จันทรา": "RN1",
    "สุขฤทัย": "RN1",
}
PN_SET = {"อรวรรณ", "มุฑิตา"}  # PN1
DOCTORS = ["พีรพัชร", "ปานหทัย", "ชัยนันท์"]  # แพทย์ 3 คน

ALLOWED_STAFF = set(RN_LEVEL_MAP.keys()) | PN_SET

# -------------------------
# 2) Helpers (ยกมาจากโค้ดเดิม)
# -------------------------
def clean_name(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    return s

def read_table_from_upload(filename: str, content: bytes) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    return pd.read_excel(io.BytesIO(content))

def count_patients_from_colA(df: pd.DataFrame) -> int:
    if df.shape[1] < 1:
        return 0
    col = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return int(col.notna().sum())

def count_doctor_entries_from_colD(df: pd.DataFrame) -> int:
    if df.shape[1] < 4:
        return 0
    col = df.iloc[:, 3].astype(str).str.strip()
    col = col.replace({"nan": "", "NaN": "", "None": ""})
    names = [clean_name(x) for x in col.tolist()]
    return int(sum(1 for n in names if n != ""))

def extract_staff_names_from_colF(df: pd.DataFrame) -> list:
    if df.shape[1] < 6:
        return []
    col = df.iloc[:, 5].astype(str).str.strip()
    col = col.replace({"nan": "", "NaN": "", "None": ""})
    names = [clean_name(x) for x in col.tolist()]
    names = [n for n in names if n != ""]
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out

def solve_rn4_base(total_revenue, n4, n3, n2, n1, npn):
    a = (n4 + n3 + n2) + 0.5*n1 + 0.5*npn
    b = (-100*n3) + (-250*n2) + 0.5*(50*n1) + 0.5*(-150*npn)
    if abs(a) < 1e-12:
        return None
    return (total_revenue - b) / a

def compute_pay_from_rn4(rn4):
    rn3 = rn4 - 100
    rn2 = rn4 - 250
    rn1 = (rn4 + 50) / 2
    pn1 = (rn4 - 150) / 2
    return rn4, rn3, rn2, rn1, pn1

def make_result_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order_role = {"RN4": 1, "RN3": 2, "RN2": 3, "RN1": 4, "PN1": 5, "DR": 6}
    df["__order"] = df["role"].map(order_role).fillna(99)
    df = df.sort_values(["__order", "name"]).drop(columns="__order")
    return df

def allocate_equal_pn(total_revenue: int, sum_rn_int: int, pn_names: list):
    npn = len(pn_names)
    if npn <= 0:
        return 0, 0
    remaining = total_revenue - sum_rn_int
    if remaining <= 0:
        return 0, 0
    pn_each = int(np.floor(remaining / npn))
    pn_total = pn_each * npn
    return pn_each, pn_total

def preview_name_levels(rn_names, pn_names):
    all_names = [clean_name(x) for x in rn_names + pn_names]
    all_names = [n for n in all_names if n]
    rows = []
    for n in all_names:
        if n in RN_LEVEL_MAP:
            rows.append({"name": n, "status": "✅ RN", "level": RN_LEVEL_MAP[n]})
        elif n in PN_SET:
            rows.append({"name": n, "status": "✅ PN", "level": "PN1"})
        else:
            rows.append({"name": n, "status": "❌ ไม่อยู่ในรายชื่อที่กำหนด", "level": ""})
    return pd.DataFrame(rows)

# -------------------------
# 3) UI (Streamlit)
# -------------------------
st.set_page_config(page_title="Dialysis Income", layout="centered")
st.title("เครื่องคำนวณรายได้ศูนย์ฟอกไต")

st.markdown("### ทางเลือก A: Upload Excel/CSV")
up = st.file_uploader("อัปโหลดไฟล์", type=["xlsx", "xls", "csv"])

st.markdown("---")
st.markdown("### ทางเลือก B: กรอกเอง (ถ้าไม่ upload)")

patients = st.number_input("จำนวนผู้ป่วย", min_value=0, step=1, value=0)

st.caption("หมายเหตุ: ระดับ RN/PN จะถูกกำหนดอัตโนมัติตามชื่อ (เช่น กาญจนา = RN4)")

default_rn = ["กาญจนา", "ทัศนีย์", "แก้วกาญจ์ตา", "จุฑารัตน์", "อัยลดา", "จันทรา"]
default_pn = ["อรวรรณ", "มุฑิตา", "", "", "", ""]

st.markdown("**พยาบาล (ใส่ชื่อได้สูงสุด 6 คน)**")
rn_inputs = []
for i in range(6):
    rn_inputs.append(st.text_input(f"พยาบาลคนที่ {i+1}", value=default_rn[i], key=f"rn{i}"))

st.markdown("**ผู้ช่วยพยาบาล (ใส่ชื่อได้สูงสุด 6 คน)**")
pn_inputs = []
for i in range(6):
    pn_inputs.append(st.text_input(f"ผู้ช่วยคนที่ {i+1}", value=default_pn[i], key=f"pn{i}"))

st.markdown("**ตรวจสอบชื่อ/ระดับ (Preview)**")
st.dataframe(preview_name_levels(rn_inputs, pn_inputs), use_container_width=True)

col1, col2 = st.columns(2)
calc = col1.button("คำนวณ", type="primary")
reset = col2.button("ล้างค่า")

if reset:
    st.session_state.clear()
    st.rerun()

# -------------------------
# 4) Calculate
# -------------------------
if calc:
    # ถ้า upload ให้ดึงค่าจากไฟล์เป็นหลัก
    doc_entries = 0
    if up is not None:
        content = up.getvalue()
        df = read_table_from_upload(up.name, content)

        patients = count_patients_from_colA(df)
        staff_names = extract_staff_names_from_colF(df)

        unknown_staff = [n for n in staff_names if n not in ALLOWED_STAFF]
        if unknown_staff:
            st.warning("พบชื่อพนักงานในคอลัมน์ F ที่ไม่อยู่ในรายชื่อที่กำหนด (จะไม่นำมาคิด): " + ", ".join(unknown_staff))

        rn_names = [n for n in staff_names if n in RN_LEVEL_MAP][:6]
        pn_names = [n for n in staff_names if n in PN_SET][:6]

        doc_entries = count_doctor_entries_from_colD(df)

        st.info(f"โหลดไฟล์สำเร็จ: ผู้ป่วย={patients}, รายการแพทย์ใน D={doc_entries}")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        rn_names = [clean_name(x) for x in rn_inputs if clean_name(x)]
        pn_names = [clean_name(x) for x in pn_inputs if clean_name(x)]

    # Validate
    errors = []
    if patients <= 0:
        errors.append("กรุณากรอกจำนวนผู้ป่วยให้มากกว่า 0 (หรือโหลดไฟล์ให้มีผู้ป่วย)")
    if len(rn_names) + len(pn_names) == 0:
        errors.append("กรุณากรอกชื่อพนักงานอย่างน้อย 1 คน")

    wrong_rn = [n for n in rn_names if n not in RN_LEVEL_MAP]
    wrong_pn = [n for n in pn_names if n not in PN_SET]
    if wrong_rn:
        errors.append("ชื่อ RN ไม่อยู่ในรายชื่อ: " + ", ".join(wrong_rn))
    if wrong_pn:
        errors.append("ชื่อ PN ไม่อยู่ในรายชื่อ: " + ", ".join(wrong_pn))

    if errors:
        st.error("พบข้อผิดพลาด:\n- " + "\n- ".join(errors))
        st.write("✅ รายชื่อที่อนุญาต")
        st.write("RN:", sorted(RN_LEVEL_MAP.keys()))
        st.write("PN:", sorted(PN_SET))
        st.stop()

    # Count levels
    levels = [RN_LEVEL_MAP[n] for n in rn_names]
    n4 = sum(1 for lv in levels if lv == "RN4")
    n3 = sum(1 for lv in levels if lv == "RN3")
    n2 = sum(1 for lv in levels if lv == "RN2")
    n1 = sum(1 for lv in levels if lv == "RN1")
    npn = len(pn_names)

    total_revenue = int(patients) * 450
    rn4_base = solve_rn4_base(total_revenue, n4, n3, n2, n1, npn)
    if rn4_base is None:
        st.error("ไม่สามารถคำนวณได้ (อาจไม่มีพนักงานจริง)")
        st.stop()

    rn4, rn3, rn2, rn1_pay, pn1 = compute_pay_from_rn4(rn4_base)

    pay_raw = {"RN4": rn4, "RN3": rn3, "RN2": rn2, "RN1": rn1_pay, "PN1": pn1}
    pay_int = {k: int(np.floor(v)) for k, v in pay_raw.items()}

    rows_staff = []
    for name in rn_names:
        lv = RN_LEVEL_MAP[name]
        rows_staff.append({"name": name, "role": lv, "pay": pay_int[lv]})

    sum_rn_int = sum(r["pay"] for r in rows_staff)
    pn_each, pn_total = allocate_equal_pn(total_revenue, sum_rn_int, pn_names)

    for name in pn_names:
        rows_staff.append({"name": name, "role": "PN1", "pay": pn_each})

    df_staff = make_result_table(rows_staff)
    staff_paid = int(df_staff["pay"].sum()) if not df_staff.empty else 0
    gap_staff = total_revenue - staff_paid

    # Doctor part
    doctor_revenue = int(doc_entries) * 350
    doc_each = int(np.floor(doctor_revenue / len(DOCTORS))) if DOCTORS else 0
    rows_doctors = [{"name": d, "role": "DR", "pay": doc_each} for d in DOCTORS]
    df_doctors = make_result_table(rows_doctors)

    # Output
    st.subheader("สรุป (รายได้จากผู้ป่วย)")
    st.write(f"จำนวนผู้ป่วย: **{int(patients)}**")
    st.write(f"รายได้รวม (ผู้ป่วย×450): **{total_revenue:,}** บาท")
    st.dataframe(df_staff, use_container_width=True)
    st.write(f"ยอดจ่ายรวมพยาบาล/ผู้ช่วย: **{staff_paid:,}** บาท")
    if gap_staff != 0:
        st.warning(f"ยอดจ่ายรวมต่ำกว่ารายได้รวม: ขาด {gap_staff:,} บาท (เพราะต้องให้ PN เท่ากันและเป็นจำนวนเต็ม)")
    else:
        st.success("ยอดจ่ายรวม = รายได้รวม (พอดี)")

    st.subheader("สรุป (รายได้แพทย์จากคอลัมน์ D)")
    st.write(f"จำนวนรายการชื่อแพทย์ในคอลัมน์ D: **{doc_entries}**")
    st.write(f"รายได้รวมแพทย์ (count×350): **{doctor_revenue:,}** บาท")
    st.dataframe(df_doctors, use_container_width=True)
