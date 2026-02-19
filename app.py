import streamlit as st
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from pyzbar.pyzbar import decode
from PIL import Image

# --- มาตราส่วนหน้ากระดาษ A5 (148mm x 210mm) ---
# ใช้ 10 พิกเซลต่อ 1 มม.
W_A5, H_A5 = 1480, 2100 

class OMRScanner:
    def __init__(self):
        self.debug_images = {}

    def preprocess(self, image):
        """1. เตรียมภาพให้พร้อมสำหรับการหาขอบ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ใช้ CLAHE ช่วยดึง Contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tile_grid_size=(8, 8))
        enhanced = clahe.apply(gray)
        # เบลอเพื่อลด Noise เล็กน้อย
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        self.debug_images['1. Grayscale (Enhanced)'] = enhanced
        return enhanced

    def detect_corners_robust(self, processed_img, original_img):
        """2. ค้นหา Corner Marks 4 มุม แบบเน้นความแม่นยำ"""
        # ใช้ Threshold แบบขาวดำสนิท
        _, thresh = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.debug_images['2. Threshold (For Corner Detection)'] = thresh
        
        # หา Contours ทั้งหมด
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # กรองเฉพาะจุดที่มีขนาดเป็นไปได้ (ไม่ใช่เศษฝุ่น)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 100: # พื้นที่ต้องใหญ่กว่า 100 px
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    candidates.append((cX, cY, area))
        
        # เรียงลำดับตามพื้นที่จากใหญ่ไปน้อย และเลือก 4 จุดที่ใหญ่ที่สุด
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)[:4]
        
        # วาดจุดที่ตรวจเจอลงบนภาพ Original เพื่อให้ User ตรวจสอบ
        debug_points_img = original_img.copy()
        pts = []
        for (x, y, a) in candidates:
            cv2.circle(debug_points_img, (x, y), 20, (0, 255, 0), -1) # วาดจุดเขียว
            pts.append([x, y])
        
        self.debug_images['3. Detected Corner Points'] = debug_points_img
        
        if len(pts) < 4:
            return None, f"พบ Corner Marks เพียง {len(pts)} จุด (ตรวจสอบแสงสว่างหรือพื้นหลัง)"
        
        # ทำการ Warp
        pts = np.array(pts, dtype="float32")
        try:
            warped = four_point_transform(original_img, pts)
            warped = cv2.resize(warped, (W_A5, H_A5))
            self.debug_images['4. Warped Result'] = warped.copy()
            return warped, None
        except Exception as e:
            return None, f"Warping Failed: {str(e)}"

    def get_omr_data(self, warped):
        """3. สแกนข้อมูล (QR & OMR)"""
        # อ่าน QR Code มุมขวาบน
        roi_qr = warped[0:500, 800:1480]
        qr_data = decode(roi_qr)
        qr_str = qr_data[0].data.decode('utf-8') if qr_data else "ไม่พบ QR Code"
        
        # --- Logic อ่านคำตอบ (จำลอง) ---
        # ในระบบจริงต้องใส่พิกัดวนลูปสแกนพิกเซลเหมือนโค้ดก่อนหน้า
        return qr_str, "000", "001"

# --- Streamlit UI ---
st.set_page_config(page_title="OMR Robust Warp", layout="wide")
st.title("🔭 OMR Answer Sheet Processor (Robust Warp Edition)")

uploaded_file = st.file_uploader("Upload Answersheet Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    scanner = OMRScanner()
    processed = scanner.preprocess(image)
    warped, error = scanner.detect_corners_robust(processed, image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🛠 Debugging Visuals")
        if error:
            st.error(error)
            st.image(scanner.debug_images.get('3. Detected Corner Points', image), caption="จุดที่ระบบมองเห็น")
            st.warning("คำแนะนำ: วางกระดาษบนพื้นโต๊ะสีตัดกัน (เช่น โต๊ะสีเข้ม) และเลี่ยงแสงสะท้อนที่หัวมุม")
        else:
            view = st.selectbox("ขั้นตอนการประมวลผล:", list(scanner.debug_images.keys()))
            st.image(scanner.debug_images[view], channels="BGR" if "Warped" in view or "Detected" in view else "RGB")

    with col2:
        if not error:
            st.subheader("📊 Extraction Results")
            qr_val, book_val, set_val = scanner.get_omr_data(warped)
            st.metric("QR Code ID", qr_val)
            st.write(f"**BookCode:** {book_val} | **SetCode:** {set_val}")
            
            st.info("💡 หากภาพ Warp ตรงแล้ว พี่บุ้งสามารถนำพิกัด OMR มาใส่ใน Module การอ่านต่อได้เลยครับ")
