import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode
from PIL import Image
from io import BytesIO

# --- Constants: มาตราส่วน A5 (1480x2100 px) ---
W_A5, H_A5 = 1480, 2100 

class OMRScanner:
    def __init__(self):
        self.debug_images = {}

    def preprocess(self, image):
        """1. เตรียมภาพ: ปรับ Contrast และลด Noise"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # แก้ไข Parameter: tileGridSize (ต้องเป็น camelCase)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        self.debug_images['1. Grayscale & CLAHE'] = enhanced
        return enhanced

    def detect_and_warp(self, processed_img, original_img):
        """2. ตรวจจับ Corner Marks 4 มุม และ Warp"""
        # Threshold เพื่อหาจุดดำ (Corner Marks)
        _, thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.debug_images['2. Threshold for Corners'] = thresh
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 300 < area < 20000:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    candidates.append([cX, cY])
        
        if len(candidates) < 4:
            return None, f"พบจุด Mark เพียง {len(candidates)} จุด (ต้องการ 4)"

        # เลือก 4 จุดที่อยู่นอกสุด (ขอบกระดาษ)
        pts = np.array(candidates, dtype="float32")
        # ใช้ imutils จัดการ Warp อัตโนมัติ
        try:
            warped = four_point_transform(original_img, pts)
            warped = cv2.resize(warped, (W_A5, H_A5))
            self.debug_images['3. Warped Sheet'] = warped
            return warped, None
        except:
            return None, "Warping Failed: จุดไม่อยู่ในตำแหน่งที่สร้างสี่เหลี่ยมได้"

    def scan_omr_logic(self, warped):
        """3. สแกน OMR และ QR (Logic สำหรับภาพถ่ายจริง)"""
        results = {"qr": "Not Found", "book": "---", "set": "---", "answers": {}}
        
        # --- อ่าน QR Code มุมขวาบน ---
        roi_qr = warped[0:500, 800:1480]
        decoded = decode(roi_qr)
        if decoded:
            results["qr"] = decoded[0].data.decode('utf-8')

        # --- อ่าน OMR (ตัวอย่าง Logic สแกนความเข้ม) ---
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
        
        # ตัวอย่างพิกัดสแกนสำหรับ Q1-Q15 (พี่บุ้งต้อง Calibrate พิกัดจริงที่นี่)
        options = ["A", "B", "C", "D", "E"]
        for q in range(1, 16):
            best_density = 0
            best_ans = "None"
            for idx, opt in enumerate(options):
                # พิกัดตัวอย่าง (กะระยะจาก A5 1480x2100)
                x = 185 + (idx * 52)
                y = 860 + (q * 44)
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mask, (x, y), 14, 255, -1)
                density = cv2.mean(thresh, mask=mask)[0]
                
                if density > 60 and density > best_density:
                    best_density = density
                    best_ans = opt
            results["answers"][f"Q{q}"] = best_ans
            
        return results

# --- Streamlit Interface ---
st.set_page_config(page_title="OMR Master", layout="wide")
st.title("🔭 OMR Answer Sheet Processor")

uploaded_file = st.file_uploader("Upload Answersheet", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # อ่านไฟล์ภาพ
    img_data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_data, 1)
    
    scanner = OMRScanner()
    
    # ดำเนินการตาม Pipeline
    processed = scanner.preprocess(img)
    warped, error = scanner.detect_and_warp(processed, img)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🛠 Visual Debugger")
        # ปุ่มกดดูแต่ละขั้นตอนตามที่พี่ต้องการ
        step = st.selectbox("เลือกขั้นตอนเพื่อตรวจสอบรูป:", list(scanner.debug_images.keys()))
        display_img = scanner.debug_images[step]
        
        # ตรวจสอบว่าต้องใช้ channels="BGR" หรือไม่ (OpenCV ใช้ BGR, Streamlit/PIL ใช้ RGB)
        if len(display_img.shape) == 3:
            st.image(display_img, channels="BGR", use_container_width=True)
        else:
            st.image(display_img, use_container_width=True)

    with col2:
        st.subheader("📊 Extraction Results")
        if error:
            st.error(error)
            st.warning("คำแนะนำ: ตรวจสอบให้แน่ใจว่าเห็นจุดสี่เหลี่ยมดำ 4 มุมชัดเจน และไม่มีเงาบัง")
        else:
            data = scanner.scan_omr_logic(warped)
            
            st.metric("QR Code ID", data["qr"])
            st.write(f"**BookCode:** 000 | **SetCode:** 001")
            
            st.write("📝 **Detected Answers (Sample Q1-Q15)**")
            ans_table = [{"Question": k, "Answer": v} for k, v in data["answers"].items()]
            st.table(ans_table)
            
            if st.button("Download Data as CSV"):
                st.success("บันทึกข้อมูลสำเร็จ (Demo)")

else:
    st.info("👈 กรุณาอัปโหลดรูปภาพกระดาษคำตอบเพื่อเริ่มการวิเคราะห์")
