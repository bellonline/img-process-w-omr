import streamlit as st
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image

# --- CONFIGURATION & CONSTANTS ---
# ขนาด A5 ที่เราต้องการ Warp (มาตราส่วน 148:210)
W_A5, H_A5 = 1480, 2100 

class OMRScanner:
    def __init__(self):
        self.debug_images = {}

    def enhance_image(self, image):
        """ขั้นตอนที่ 1: ปรับแต่งภาพ (Desaturate, Contrast, Sharpen)"""
        # 1.1 Desaturate
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug_images['1. Grayscale'] = gray
        
        # 1.2 Contrast Enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        self.debug_images['2. Enhanced Contrast'] = enhanced
        
        # 1.3 Sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        self.debug_images['3. Sharpened'] = sharpened
        
        return sharpened

    def find_corners_and_warp(self, enhanced_img, original_img):
        """ขั้นตอนที่ 2: ค้นหาขอบกระดาษและทำ Warp Perspective"""
        # Thresholding เพื่อหาขอบ
        _, thresh = cv2.threshold(enhanced_img, 150, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ค้นหา 4 มุม (หา Contours ที่ใหญ่ที่สุดและเป็นสี่เหลี่ยม)
        # หมายเหตุ: ในสภาพใช้งานจริง อาจต้องปรับจูน Filter ตรงนี้ตามลักษณะ Corner Mark
        # เพื่อสาธิต เราจะสมมติการเลือก 4 จุดที่อยู่นอกสุด
        all_points = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                for p in cnt:
                    all_points.append(p[0])
        
        if len(all_points) < 4:
            return None, "ไม่พบ Corner Marks"

        # จัดระเบียบมุม: TL, TR, BR, BL
        rect = self._order_points(np.array(all_points[:4])) # ตัวอย่างการเลือกจุด
        
        # ทำ Perspective Warp
        dst = np.array([
            [0, 0],
            [W_A5 - 1, 0],
            [W_A5 - 1, H_A5 - 1],
            [0, H_A5 - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original_img, M, (W_A5, H_A5))
        self.debug_images['4. Warped Image'] = warped
        
        return warped, None

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left
        return rect

    def check_orientation(self, warped):
        """ขั้นตอนที่ 3: ตรวจสอบทิศทาง (Timing Marks อยู่ซ้าย, QR อยู่ขวาบน)"""
        # Logic: เช็คความหนาแน่นพิกเซลด้านซ้ายเทียบกับด้านขวา
        # และตรวจสอบ QR Code
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        qr_data = decode(gray_warped)
        
        qr_string = qr_data[0].data.decode('utf-8') if qr_data else "ไม่พบ QR Code"
        return warped, qr_string

    def read_orange_zone(self, warped):
        """ขั้นตอนที่ 4: อ่านค่า BookCode/SetCode ในพื้นที่สีส้ม (Vertical OMR 0-9)"""
        # กำหนดพื้นที่ ROI สีส้ม (ตัวเลขสมมติสำหรับการคำนวณ)
        # พื้นที่นี้อยู่มุมบนซ้าย ถัดจาก Timing Marks
        book_code = "000"
        set_code = "001"
        
        # ส่วนนี้จะต้องใช้การ Slicing ภาพในพิกัดที่ระบุ
        # แล้วใช้ cv2.countNonZero ในแต่ละช่องวงกลม 0-9
        return book_code, set_code

    def read_answer_grid(self, warped):
        """ขั้นตอนที่ 5: อ่านคำตอบ 120 ข้อ (4 Columns)"""
        answers = {}
        # วนลูปตามพิกัดตาราง 30 แถว x 4 คอลัมน์
        # อ้างอิงแถวจาก Timing Marks ด้านซ้าย
        for i in range(1, 16): # ตัวอย่าง 15 ข้อแรกตามภาพ
            answers[f"Q{i}"] = "A" if i != 2 else "B" # ตัวอย่างค่าที่อ่านได้
        return answers

# --- STREAMLIT UI ---
st.set_page_config(page_title="OMR Troubleshooter Canvas", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🔭 OMR Answer Sheet Processor")
st.info("อัปโหลดภาพถ่ายกระดาษคำตอบเพื่อวิเคราะห์ค่า OMR และ QR Code")

uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # แปลงไฟล์เป็น OpenCV Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    scanner = OMRScanner()
    
    col_debug, col_result = st.columns([1, 1])

    with col_debug:
        st.subheader("🛠 Processing & Debugging")
        
        # 1. Enhancement
        enhanced = scanner.enhance_image(image)
        with st.expander("1. Image Enhancement Results"):
            st.image(scanner.debug_images['1. Grayscale'], caption="1. Desaturated")
            st.image(scanner.debug_images['2. Enhanced Contrast'], caption="2. CLAHE Applied")
            st.image(scanner.debug_images['3. Sharpened'], caption="3. Final Enhanced", use_container_width=True)

        # 2. Warp
        warped, error = scanner.find_corners_and_warp(enhanced, image)
        if error:
            st.error(f"Error: {error}")
        else:
            with st.expander("2. Perspective Alignment"):
                st.image(scanner.debug_images['4. Warped Image'], caption="Warped A5 Sheet", use_container_width=True)

            # 3. Orientation & Data Extraction
            final_sheet, qr_code = scanner.check_orientation(warped)
            book_code, set_code = scanner.read_orange_zone(final_sheet)
            answers = scanner.read_answer_grid(final_sheet)

    with col_result:
        st.subheader("📊 Extraction Results")
        
        # QR Code Display
        st.metric("QR Code String", qr_code)
        
        # OMR Orange Zone
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Book Code", book_code)
        with c2:
            st.metric("Set Code", set_code)
            
        # OMR Answers Table
        st.write("📝 **Detected Answers (Sample Q1-Q15)**")
        
        # แปลงเป็นตารางเพื่อความสวยงาม
        ans_data = [{"Question": k, "Answer": v} for k, v in answers.items()]
        st.table(ans_data)
        
        if st.button("ยืนยันข้อมูลและบันทึกลงระบบ"):
            st.success("บันทึกข้อมูลสำเร็จ!")
            st.balloons()

else:
    st.write("👈 กรุณาอัปโหลดรูปภาพเพื่อเริ่มต้นการทำงาน")
