import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode
from PIL import Image

# --- Constants ---
W_A5, H_A5 = 1480, 2100 # มาตราส่วน 10px : 1mm

class OMRScanner:
    def __init__(self):
        self.debug_images = {}

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        self.debug_images['1. Preprocessed'] = blurred
        return blurred

    def detect_and_warp(self, processed_img, original_img):
        # หาขอบเพื่อจับ Corner Marks
        thresh = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        centers = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 400 < area < 15000: # ขนาด Corner Mark
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
        
        if len(centers) < 4:
            return None, f"พบ Corner Marks เพียง {len(centers)} จุด (ต้องการ 4)"

        # Warp Perspective ด้วย imutils
        pts = np.array(centers[:4], dtype="float32")
        warped = four_point_transform(original_img, pts)
        warped = cv2.resize(warped, (W_A5, H_A5))
        self.debug_images['2. Warped'] = warped.copy()
        return warped, None

    def fix_orientation_and_qr(self, warped):
        """หมุนภาพและอ่าน QR"""
        for i in range(4):
            # ตรวจสอบ QR ในพื้นที่มุมขวาบน (ROI)
            roi_qr = warped[0:500, 800:1480]
            decoded = decode(roi_qr)
            if decoded:
                return warped, decoded[0].data.decode('utf-8')
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped, "QR Not Found"

    def scan_omr(self, warped):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
        
        # ภาพสำหรับ Debug (วาดวงกลมจุดที่สแกน)
        debug_view = warped.copy()
        
        # --- Logic อ่าน Header (พื้นที่สีส้ม) ---
        # สมมติพิกัด BookCode (ต้องปรับตามกระดาษจริง)
        book_code = ""
        for col in range(3):
            best_val = 0
            selected = 0
            for row in range(10):
                x, y = 140 + (col * 45), 200 + (row * 38)
                cv2.circle(debug_view, (x, y), 10, (0, 0, 255), 2) # วาดจุด Debug
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mask, (x, y), 12, 255, -1)
                mean = cv2.mean(thresh, mask=mask)[0]
                if mean > best_val:
                    best_val = mean
                    selected = row
            book_code += str(selected)

        # --- Logic อ่านคำตอบ 120 ข้อ (ตัวอย่าง Column 1) ---
        options = ["A", "B", "C", "D", "E"]
        answers = {}
        for q in range(1, 16):
            best_val = 0
            ans = "None"
            for idx, opt in enumerate(options):
                # ปรับพิกัด X, Y ตรงนี้ให้ตรงกับวงกลมในกระดาษ
                x = 185 + (idx * 52)
                y = 860 + (q * 44)
                cv2.circle(debug_view, (x, y), 10, (255, 0, 0), 2) # วาดจุด Debug (น้ำเงิน)
                
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.circle(mask, (x, y), 14, 255, -1)
                density = cv2.mean(thresh, mask=mask)[0]
                if density > 60 and density > best_val:
                    best_val = density
                    ans = opt
            answers[f"Q{q}"] = ans
            
        self.debug_images['3. Scan Area Check'] = debug_view
        return book_code, answers

# --- Streamlit UI ---
st.set_page_config(page_title="OMR Pro Troubleshooter", layout="wide")
st.title("🔭 OMR Answer Sheet Processor")

uploaded_file = st.file_uploader("Upload Answersheet", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    scanner = OMRScanner()
    processed = scanner.preprocess(image)
    warped, error = scanner.detect_and_warp(processed, image)
    
    if error:
        st.error(f"❌ {error}")
    else:
        final_sheet, qr_code = scanner.fix_orientation_and_qr(warped)
        book_code, answers = scanner.scan_omr(final_sheet)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼 Debugging View")
            mode = st.radio("ดูภาพขั้นตอน:", ["1. Preprocessed", "2. Warped", "3. Scan Area Check"])
            st.image(scanner.debug_images[mode], channels="BGR" if "Scan" in mode or "Warp" in mode else "RGB")
            st.caption("วงกลมสีแดง/น้ำเงิน คือจุดที่ระบบพยายามอ่านค่าพิกเซล")

        with col2:
            st.subheader("📊 Extraction Results")
            st.metric("QR Code ID", qr_code)
            st.metric("Book Code", book_code)
            
            # แสดงตารางเปรียบเทียบ
            st.table([{"Question": k, "Answer": v} for k, v in answers.items()])
