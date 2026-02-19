import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode
from PIL import Image

# --- ค่าคงที่สำหรับการประมวลผล ---
W_A5, H_A5 = 1480, 2100  # อัตราส่วน A5 (10px ต่อ 1mm)

class OMRScanner:
    def __init__(self):
        self.debug_images = {}

    def preprocess(self, image):
        """1. ปรับแต่งภาพเบื้องต้น (Desaturate & Enhance)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ปรับ Contrast ด้วย CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # ลด Noise เล็กน้อย
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        self.debug_images['1. Preprocessed'] = blurred
        return blurred

    def detect_and_warp(self, processed_img, original_img):
        """2. ตรวจจับ Corner Marks 4 มุม และ Warp ด้วย imutils"""
        # Threshold เพื่อหาวัตถุสีดำ (Corner Marks)
        thresh = cv2.adaptiveThreshold(processed_img, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        centers = []
        # ค้นหา Contours ที่มีลักษณะเป็นสี่เหลี่ยมมุมกระดาษ
        for c in cnts:
            area = cv2.contourArea(c)
            if 400 < area < 10000: # ปรับขนาดตามความละเอียดภาพถ่าย
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4: # เป็นรูป 4 เหลี่ยม
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centers.append((cX, cY))
        
        if len(centers) < 4:
            return None, f"พบ Corner Marks เพียง {len(centers)} จุด (ต้องการ 4)"

        # ใช้ imutils ในการ Warp (จัดการการเรียงจุดให้อัตโนมัติ)
        pts = np.array(centers, dtype="float32")
        warped = four_point_transform(original_img, pts)
        
        # Resize ให้เป็นขนาด A5 มาตรฐาน
        warped = cv2.resize(warped, (W_A5, H_A5))
        self.debug_images['2. Warped'] = warped
        return warped, None

    def fix_orientation(self, warped):
        """3. หมุนภาพให้ถูกทิศทาง (Timing Marks ซ้าย, QR ขวาบน)"""
        # ลองหมุนภาพ 4 ทิศทาง เพื่อหาจุดที่ QR Code อยู่มุมขวาบน
        for i in range(4):
            # ตรวจสอบ QR ในพื้นที่มุมขวาบน
            roi_qr = warped[0:500, 800:W_A5]
            decoded = decode(roi_qr)
            if decoded:
                self.debug_images['3. Orientation Fixed'] = warped
                return warped, decoded[0].data.decode('utf-8')
            
            # ถ้าไม่เจอให้หมุน 90 องศา
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            
        return warped, "QR Not Found"

    def scan_omr(self, warped):
        """4. สแกนพื้นที่ OMR (Orange Zone & Answers)"""
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Threshold สำหรับอ่านรอยฝน
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 25, 12)
        
        results = {"BookCode": "000", "SetCode": "001", "Answers": {}}
        
        # --- ตัวอย่าง Logic การสแกนคำตอบ (Calibrate พิกัดที่นี่) ---
        # พี่บุ้งต้องวัดระยะจากภาพ Warped (1480x2100) แล้วใส่พิกัดจริง
        options = ["A", "B", "C", "D", "E"]
        for q in range(1, 16):
            darkest_val = 0
            best_choice = "None"
            for idx, opt in enumerate(options):
                # พิกัดจำลอง (X, Y) สำหรับแต่ละวงกลม
                x = 185 + (idx * 52)
                y = 825 + (q * 44)
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mask, (x, y), 15, 255, -1)
                mean = cv2.mean(thresh, mask=mask)[0]
                
                if mean > 60 and mean > darkest_val:
                    darkest_val = mean
                    best_choice = opt
            results["Answers"][f"Q{q}"] = best_choice
            
        return results

# --- Streamlit UI ---
st.set_page_config(page_title="OMR Imutils Pro", layout="wide")
st.title("🔭 OMR Answer Sheet Processor (imutils version)")

# อย่าลืมบอกให้ User อัปเดต requirements.txt
with st.sidebar:
    st.header("Settings")
    st.info("ใช้ imutils.four_point_transform ในการ Warp")

uploaded_file = st.file_uploader("อัปโหลดภาพกระดาษคำตอบ", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # อ่านภาพ
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    scanner = OMRScanner()
    
    # 1. Preprocess
    processed = scanner.preprocess(image)
    
    # 2. Detect & Warp
    warped, error = scanner.detect_and_warp(processed, image)
    
    if error:
        st.error(f"❌ {error}")
        st.image(processed, caption="ผลการหาขอบภาพ (กรุณาถ่ายให้เห็น Corner Marks ชัดเจน)")
    else:
        # 3. Fix Orientation & Read QR
        final_sheet, qr_string = scanner.fix_orientation(warped)
        
        # 4. Scan Data
        data = scanner.scan_omr(final_sheet)
        
        # --- แสดงผล ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼 Processing Steps")
            st.image(scanner.debug_images['1. Preprocessed'], caption="1. Contrast Enhanced")
            st.image(final_sheet, caption="2. Warped & Rotated Sheet", channels="BGR")
            
        with col2:
            st.subheader("📊 Extraction Results")
            st.metric("QR Code ID", qr_string)
            
            c1, c2 = st.columns(2)
            c1.metric("Book Code", data["BookCode"])
            c2.metric("Set Code", data["SetCode"])
            
            st.write("📝 **Answer Grid (Preview Q1-Q15)**")
            ans_table = [{"Question": k, "Answer": v} for k, v in data["Answers"].items()]
            st.table(ans_table)
            
            if st.button("Download Data as JSON"):
                st.json(data)
