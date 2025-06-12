import cv2
import easyocr
import sys
import os
import numpy as np
import time

# Global reader để tránh khởi tạo lại
reader = None

def init_ocr():
    """Khởi tạo OCR một lần duy nhất"""
    global reader
    if reader is None:
        print("⚡ Khởi tạo EasyOCR (chỉ một lần)...")
        start_time = time.time()
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print(f"✅ OCR ready ({time.time() - start_time:.1f}s)")
    return reader

def fast_ocr_test(image_path):
    """Test OCR siêu nhanh"""
    print(f"🚀 FAST OCR: {os.path.basename(image_path)}")
    
    if not os.path.exists(image_path):
        print(f"❌ File không tồn tại")
        return
    
    # Khởi tạo OCR
    ocr_reader = init_ocr()
    
    # Đọc ảnh
    start_time = time.time()
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Không đọc được ảnh")
        return
    
    # Tối ưu kích thước
    height, width = img.shape[:2]
    original_size = width * height
    
    # Resize nhanh nếu ảnh quá lớn
    if width > 800 or height > 800:
        scale = min(800/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        print(f"📏 Resize: {width}x{height} → {new_width}x{new_height}")
    
    # OCR siêu nhanh
    print("⚡ OCR processing...")
    ocr_start = time.time()
    results = ocr_reader.readtext(img, paragraph=False, width_ths=0.8, height_ths=0.8)
    ocr_time = time.time() - ocr_start
    
    # Lọc kết quả nhanh
    license_plates = []
    if results:
        for bbox, text, conf in results:
            if conf > 0.5:  # Chỉ lấy confidence cao
                clean_text = text.replace(" ", "").replace("-", "").upper()
                if 4 <= len(clean_text) <= 10:
                    has_digit = any(c.isdigit() for c in clean_text)
                    has_letter = any(c.isalpha() for c in clean_text)
                    if has_digit and has_letter:
                        license_plates.append((clean_text, conf))
    
    total_time = time.time() - start_time
    
    # Kết quả
    print(f"⏱️ Thời gian: {total_time:.2f}s (OCR: {ocr_time:.2f}s)")
    
    if license_plates:
        print("🎯 BIỂN SỐ:")
        for plate, conf in license_plates:
            print(f"   • {plate} ({conf:.3f})")
    else:
        print("❌ Không tìm thấy biển số")
    
    # Hiển thị nhanh (không vẽ phức tạp)
    if license_plates:
        display_img = img.copy()
        
        # Vẽ text đơn giản
        for i, (plate, conf) in enumerate(license_plates):
            y_pos = 30 + i * 40
            cv2.putText(display_img, f"{plate} ({conf:.2f})", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Resize để hiển thị
        if display_img.shape[1] > 800:
            scale = 800 / display_img.shape[1]
            new_w = int(display_img.shape[1] * scale)
            new_h = int(display_img.shape[0] * scale)
            display_img = cv2.resize(display_img, (new_w, new_h))
        
        cv2.imshow('Fast OCR', display_img)
        print("📋 Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def batch_test():
    """Test nhiều ảnh liên tục"""
    print("🚀 BATCH FAST OCR TEST")
    print("=" * 40)
    
    # Tìm ảnh test
    test_dirs = ["dataset/images/val", "dataset/images/train", "test_results"]
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            test_images.extend([os.path.join(test_dir, img) for img in images[:3]])
    
    if not test_images:
        print("❌ Không tìm thấy ảnh test")
        return
    
    print(f"📊 Test {len(test_images)} ảnh...")
    
    # Khởi tạo OCR một lần
    init_ocr()
    
    total_start = time.time()
    success_count = 0
    
    for i, img_path in enumerate(test_images, 1):
        print(f"\n📷 {i}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # Test nhanh không hiển thị GUI
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Resize nhanh
                if img.shape[1] > 600:
                    scale = 600 / img.shape[1]
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (new_w, new_h))
                
                # OCR nhanh
                results = reader.readtext(img, paragraph=False)
                
                # Tìm biển số
                found_plates = []
                for bbox, text, conf in results:
                    if conf > 0.5:
                        clean = text.replace(" ", "").replace("-", "").upper()
                        if 4 <= len(clean) <= 10:
                            if any(c.isdigit() for c in clean) and any(c.isalpha() for c in clean):
                                found_plates.append(clean)
                
                if found_plates:
                    success_count += 1
                    print(f"   ✅ {', '.join(found_plates)}")
                else:
                    print(f"   ❌ Không tìm thấy")
    
    total_time = time.time() - total_start
    avg_time = total_time / len(test_images)
    
    print(f"\n📊 KẾT QUẢ:")
    print(f"   • Tổng thời gian: {total_time:.1f}s")
    print(f"   • Trung bình: {avg_time:.2f}s/ảnh")
    print(f"   • Thành công: {success_count}/{len(test_images)} ({success_count/len(test_images)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            batch_test()
        else:
            fast_ocr_test(sys.argv[1])
    else:
        print("🚀 FAST OCR - Tối ưu tốc độ")
        print("=" * 30)
        print("Cách dùng:")
        print("  python fast_ocr.py <ảnh>     - Test 1 ảnh")
        print("  python fast_ocr.py batch     - Test nhiều ảnh")
        print()
        print("Ví dụ:")
        print("  python fast_ocr.py kq.jpg")
        print("  python fast_ocr.py batch")
