from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import mysql.connector, os, csv, io, json
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AttendTrack API")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Update with your Vercel URL after deploy e.g. "https://attendtrack.vercel.app"
    allow_methods=["*"], allow_headers=["*"])

def get_db():
    config = dict(
        host=os.getenv("DB_HOST","gateway01.ap-southeast-1.prod.alicloud.tidbcloud.com"),
        port=int(os.getenv("DB_PORT","4000")),
        user=os.getenv("DB_USER","dFJR6gPxogDgfwt.root"),
        password=os.getenv("DB_PASSWORD","FquQuTSKnO1mCkdI"),
        database=os.getenv("DB_NAME","attendtrack"),
    )
    # TiDB requires SSL — set DB_SSL_CA=true in Render env vars
    if os.getenv("DB_SSL_CA"):
        config["ssl_ca"]             = os.getenv("DB_SSL_CA")
        config["ssl_verify_cert"]    = True
        config["ssl_verify_identity"] = True
    return mysql.connector.connect(**config)

@app.on_event("startup")
def init_tables():
    try:
        db = get_db(); cur = db.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS employees (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            full_name       VARCHAR(150) NOT NULL,
            email           VARCHAR(255) UNIQUE NOT NULL,
            aadhaar_no      VARCHAR(12)  UNIQUE NOT NULL,
            department      VARCHAR(100) NOT NULL,
            shift_hrs       DECIMAL(4,1) DEFAULT 8.0,
            face_descriptor JSON,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS attendance (
            id        INT AUTO_INCREMENT PRIMARY KEY,
            emp_id    INT NOT NULL,
            date      DATE NOT NULL,
            clock_in  TIME,
            clock_out TIME,
            total_hrs DECIMAL(5,2) DEFAULT 0,
            ot_hrs    DECIMAL(5,2) DEFAULT 0,
            status    ENUM('on-duty','present','absent') DEFAULT 'absent',
            UNIQUE KEY unique_emp_date (emp_id, date),
            FOREIGN KEY (emp_id) REFERENCES employees(id) ON DELETE CASCADE)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS admin_settings (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            pay_per_day      DECIMAL(10,2) DEFAULT 500.00,
            ot_pay_per_hr    DECIMAL(10,2) DEFAULT 100.00,
            food_allowance   DECIMAL(10,2) DEFAULT 50.00,
            food_before_time VARCHAR(5)    DEFAULT '08:00',
            tds_amount       DECIMAL(10,2) DEFAULT 13.00,
            updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)""")
        # Add new columns if upgrading from old schema
        for col, defn in [
            ("food_allowance",   "DECIMAL(10,2) DEFAULT 50.00"),
            ("food_before_time", "VARCHAR(5)    DEFAULT '08:00'"),
            ("tds_amount",       "DECIMAL(10,2) DEFAULT 13.00"),
        ]:
            try:
                cur.execute(f"ALTER TABLE admin_settings ADD COLUMN {col} {defn}")
            except: pass
        cur.execute("""INSERT IGNORE INTO admin_settings
            (id,pay_per_day,ot_pay_per_hr,food_allowance,food_before_time,tds_amount)
            VALUES (1,500.00,100.00,50.00,'08:00',13.00)""")
        db.commit(); db.close()
        print("Tables ready")
    except Exception as e:
        print(f"Startup warning: {e}")

def row_to_dict(cursor, row):
    return dict(zip([c[0] for c in cursor.description], row))

def time_to_str(t):
    if t is None: return None
    if isinstance(t, str): return t[:5]
    if hasattr(t, "seconds"):
        total = int(t.total_seconds())
        h, rem = divmod(total, 3600)
        return f"{h:02d}:{rem//60:02d}"
    return str(t)[:5]

def time_before(t_str, limit_str):
    """Returns True if t_str (HH:MM) is before limit_str (HH:MM)."""
    if not t_str: return False
    return t_str <= limit_str

# ── Models ──────────────────────────────────────────────────────────────────
class EmployeeCreate(BaseModel):
    full_name: str
    email: str
    aadhaar_no: str
    department: str
    shift_hrs: float = 8.0
    face_descriptor: Optional[List[float]] = None

class FaceDescriptorUpdate(BaseModel):
    face_descriptor: List[float]

class ClockAction(BaseModel):
    emp_id: int

class SettingsUpdate(BaseModel):
    pay_per_day: float
    ot_pay_per_hr: float
    food_allowance: float
    food_before_time: str
    tds_amount: float

# ── EMPLOYEES ───────────────────────────────────────────────────────────────
@app.get("/employees")
def list_employees():
    db = get_db(); cur = db.cursor()
    cur.execute("SELECT id,full_name,email,aadhaar_no,department,shift_hrs,created_at FROM employees ORDER BY full_name")
    result = [row_to_dict(cur, r) for r in cur.fetchall()]
    for r in result: r["created_at"] = str(r["created_at"])
    db.close(); return result

@app.get("/employees/faces/all")
def get_all_faces():
    db = get_db(); cur = db.cursor()
    cur.execute("SELECT id,full_name,face_descriptor FROM employees WHERE face_descriptor IS NOT NULL")
    result = []
    for row in cur.fetchall():
        fd = row[2]
        if isinstance(fd, str): fd = json.loads(fd)
        result.append({"id": row[0], "full_name": row[1], "face_descriptor": fd})
    db.close(); return result

@app.get("/employees/{emp_id}")
def get_employee(emp_id: int):
    db = get_db(); cur = db.cursor()
    cur.execute("SELECT id,full_name,email,aadhaar_no,department,shift_hrs,created_at FROM employees WHERE id=%s", (emp_id,))
    row = cur.fetchone()
    if not row: db.close(); raise HTTPException(404,"Employee not found")
    d = row_to_dict(cur, row); d["created_at"] = str(d["created_at"])
    db.close(); return d

@app.post("/employees", status_code=201)
def create_employee(emp: EmployeeCreate):
    db = get_db(); cur = db.cursor()
    try:
        fd = json.dumps(emp.face_descriptor) if emp.face_descriptor else None
        cur.execute(
            "INSERT INTO employees (full_name,email,aadhaar_no,department,shift_hrs,face_descriptor) VALUES (%s,%s,%s,%s,%s,%s)",
            (emp.full_name, emp.email, emp.aadhaar_no, emp.department, emp.shift_hrs, fd))
        db.commit(); new_id = cur.lastrowid
    except mysql.connector.IntegrityError as e:
        db.close(); raise HTTPException(409, str(e))
    db.close(); return {"message":"Employee registered","id": new_id}

@app.put("/employees/{emp_id}/face")
def update_face(emp_id: int, body: FaceDescriptorUpdate):
    db = get_db(); cur = db.cursor()
    cur.execute("UPDATE employees SET face_descriptor=%s WHERE id=%s",
                (json.dumps(body.face_descriptor), emp_id))
    db.commit(); db.close(); return {"message":"Face descriptor saved"}

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int):
    db = get_db(); cur = db.cursor()
    cur.execute("DELETE FROM employees WHERE id=%s", (emp_id,))
    db.commit(); db.close(); return {"message":"Removed"}

# ── CLOCK IN / OUT ──────────────────────────────────────────────────────────
@app.post("/clock")
def clock(action: ClockAction):
    db = get_db(); cur = db.cursor()
    today    = date.today().isoformat()
    now_time = datetime.now().strftime("%H:%M")
    cur.execute("SELECT * FROM employees WHERE id=%s", (action.emp_id,))
    row = cur.fetchone()
    if not row: db.close(); raise HTTPException(404,"Employee not found")
    emp = row_to_dict(cur, row); shift = float(emp["shift_hrs"])
    cur.execute("SELECT * FROM attendance WHERE emp_id=%s AND date=%s", (action.emp_id, today))
    att_row = cur.fetchone()
    if att_row is None:
        cur.execute("INSERT INTO attendance (emp_id,date,clock_in,status) VALUES (%s,%s,%s,'on-duty')",
                    (action.emp_id, today, now_time))
        db.commit(); db.close()
        return {"action":"clock_in","time":now_time,"emp_name":emp["full_name"]}
    att = row_to_dict(cur, att_row)
    ci = time_to_str(att["clock_in"]); co = time_to_str(att["clock_out"])
    if ci and not co:
        in_min  = int(ci[:2])*60 + int(ci[3:5])
        out_min = int(now_time[:2])*60 + int(now_time[3:5])
        total   = round((out_min - in_min)/60, 2)
        ot      = round(max(0, total - shift), 2)
        cur.execute("""UPDATE attendance SET clock_out=%s,total_hrs=%s,ot_hrs=%s,status='present'
                       WHERE emp_id=%s AND date=%s""",
                    (now_time, total, ot, action.emp_id, today))
        db.commit(); db.close()
        return {"action":"clock_out","time":now_time,"total_hrs":total,"ot_hrs":ot,"emp_name":emp["full_name"]}
    db.close(); raise HTTPException(400,"Already completed attendance today")

# ── ATTENDANCE ──────────────────────────────────────────────────────────────
@app.get("/attendance")
def get_attendance(emp_id: Optional[int]=None, month: Optional[str]=None, date_filter: Optional[str]=None):
    db = get_db(); cur = db.cursor()
    q = """SELECT a.*,e.full_name,e.email,e.aadhaar_no,e.department,e.shift_hrs
           FROM attendance a JOIN employees e ON a.emp_id=e.id WHERE 1=1"""
    p = []
    if emp_id:      q += " AND a.emp_id=%s";                       p.append(emp_id)
    if month:       q += " AND DATE_FORMAT(a.date,'%%Y-%%m')=%s";  p.append(month)
    if date_filter: q += " AND a.date=%s";                         p.append(date_filter)
    q += " ORDER BY a.date DESC,e.full_name"
    cur.execute(q, p)
    result = []
    for r in cur.fetchall():
        d = row_to_dict(cur, r)
        d["clock_in"]  = time_to_str(d["clock_in"])
        d["clock_out"] = time_to_str(d["clock_out"])
        d["date"]      = str(d["date"])
        result.append(d)
    db.close(); return result

@app.get("/attendance/today")
def get_today():
    return get_attendance(date_filter=date.today().isoformat())

# ── DASHBOARD ───────────────────────────────────────────────────────────────
@app.get("/dashboard")
def dashboard():
    db = get_db(); cur = db.cursor()
    today = date.today().isoformat()
    cur.execute("SELECT COUNT(*) FROM employees")
    total = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(*) FROM attendance WHERE date=%s AND status IN ('present','on-duty')",(today,))
    present = cur.fetchone()[0] or 0
    cur.execute("""SELECT COUNT(*) FROM employees WHERE id NOT IN
        (SELECT emp_id FROM attendance WHERE date=%s AND status IN ('present','on-duty'))""",(today,))
    absent = cur.fetchone()[0] or 0
    cur.execute("SELECT COALESCE(SUM(ot_hrs),0) FROM attendance WHERE date=%s",(today,))
    ot = float(cur.fetchone()[0] or 0)
    cur.execute("""SELECT a.emp_id,a.clock_in,e.full_name FROM attendance a
        JOIN employees e ON a.emp_id=e.id WHERE a.date=%s AND a.status='on-duty'""",(today,))
    on_duty = [{"emp_id":r[0],"clock_in":time_to_str(r[1]),"name":r[2]} for r in cur.fetchall()]
    cur.execute("""SELECT id,full_name,department FROM employees WHERE id NOT IN
        (SELECT emp_id FROM attendance WHERE date=%s AND status IN ('present','on-duty'))""",(today,))
    absent_list = [{"id":r[0],"name":r[1],"dept":r[2]} for r in cur.fetchall()]
    db.close()
    return {"total_employees":total,"present":present,"absent":absent,
            "ot_hours":round(ot,1),"on_duty":on_duty,"absent_employees":absent_list}

# ── ADMIN SETTINGS ──────────────────────────────────────────────────────────
@app.get("/settings")
def get_settings():
    db = get_db(); cur = db.cursor()
    cur.execute("SELECT pay_per_day,ot_pay_per_hr,food_allowance,food_before_time,tds_amount FROM admin_settings WHERE id=1")
    row = cur.fetchone(); db.close()
    if not row: return {"pay_per_day":500.0,"ot_pay_per_hr":100.0,"food_allowance":50.0,"food_before_time":"08:00","tds_amount":13.0}
    return {"pay_per_day":float(row[0]),"ot_pay_per_hr":float(row[1]),
            "food_allowance":float(row[2]),"food_before_time":row[3],"tds_amount":float(row[4])}

@app.put("/settings")
def update_settings(s: SettingsUpdate):
    db = get_db(); cur = db.cursor()
    cur.execute("""UPDATE admin_settings
        SET pay_per_day=%s,ot_pay_per_hr=%s,food_allowance=%s,food_before_time=%s,tds_amount=%s
        WHERE id=1""",
        (s.pay_per_day, s.ot_pay_per_hr, s.food_allowance, s.food_before_time, s.tds_amount))
    db.commit(); db.close(); return {"message":"Settings updated"}

# ── REPORTS CSV ─────────────────────────────────────────────────────────────
@app.get("/reports/csv")
def report_csv(month: Optional[str]=None, date_filter: Optional[str]=None, emp_id: Optional[int]=None):
    if not month and not date_filter:
        raise HTTPException(400,"Provide month or date_filter")
    records  = get_attendance(emp_id=emp_id, month=month, date_filter=date_filter)
    s        = get_settings()
    ppd      = s["pay_per_day"]
    otp      = s["ot_pay_per_hr"]
    food_amt = s["food_allowance"]
    food_cut = s["food_before_time"]
    tds_amt  = s["tds_amount"]

    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["Name","Email","Aadhaar","Department","Date","Clock In","Clock Out",
                "Total Hrs","OT Hrs","Status",
                "Day Pay (Rs)","OT Pay (Rs)","Food Allowance (Rs)",
                "Gross (Rs)","TDS Deducted (Rs)","Net Pay (Rs)"])
    for r in records:
        day_pay  = ppd if r["status"] == "present" else 0
        ot_pay   = round(float(r["ot_hrs"] or 0) * otp, 2)
        food     = food_amt if (r["status"] in ("present","on-duty") and time_before(r["clock_in"], food_cut)) else 0
        gross    = round(day_pay + ot_pay + food, 2)
        tds      = tds_amt if day_pay > 0 else 0   # flat rupee deduction per present day
        net      = round(gross - tds, 2)
        w.writerow([r["full_name"],r["email"],r["aadhaar_no"],r["department"],r["date"],
                    r["clock_in"] or "",r["clock_out"] or "",r["total_hrs"],r["ot_hrs"],r["status"],
                    day_pay, ot_pay, food, gross, tds, net])
    out.seek(0)
    label = date_filter or month
    return StreamingResponse(iter([out.getvalue()]),media_type="text/csv",
        headers={"Content-Disposition":f"attachment; filename=attendance_{label}.csv"})
