import streamlit as st
import pytesseract
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import concurrent.futures

# Set page config for a wider layout
st.set_page_config(page_title="Bulk URL Scanner", layout="wide")

def preprocess_image_for_ocr(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    gray_resized = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [gray, gray_resized, thresh]

def multi_scale_template_match(target_gray, logo_gray, threshold=0.7):
    logo_h, logo_w = logo_gray.shape[:2]
    for scale in np.linspace(0.2, 2.0, 20):
        resized_w = int(logo_w * scale)
        resized_h = int(logo_h * scale)
        
        if resized_h > target_gray.shape[0] or resized_w > target_gray.shape[1]:
            continue
            
        resized_logo = cv2.resize(logo_gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(target_gray, resized_logo, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= threshold:
            return True
    return False

def process_single_url(url, logo_gray, target_word):
    word_found = "Yes" if target_word.lower() in url.lower() else "No"
    logo_found = "No"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() 
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'image' in content_type:
            image_bytes = response.content
            nparr = np.frombuffer(image_bytes, np.uint8)
            cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if cv_img is not None:
                preprocessed_imgs = preprocess_image_for_ocr(cv_img)
                pil_raw = Image.open(BytesIO(image_bytes))
                
                text_candidates = [
                    pytesseract.image_to_string(pil_raw), 
                    pytesseract.image_to_string(pil_raw, config='--psm 11'),
                    pytesseract.image_to_string(pil_raw, config='--psm 6')
                ]
                
                for p_img in preprocessed_imgs:
                    text_candidates.append(pytesseract.image_to_string(p_img))
                    text_candidates.append(pytesseract.image_to_string(p_img, config='--psm 11'))
                    text_candidates.append(pytesseract.image_to_string(p_img, config='--psm 6'))
                
                full_text = " ".join(text_candidates).lower()
                
                if target_word.lower() in full_text:
                    word_found = "Yes"
                
                if logo_gray is not None:
                    target_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    if multi_scale_template_match(target_gray, logo_gray, threshold=0.7):
                        logo_found = "Yes"
            else:
                word_found = "Error Decoding"
                logo_found = "Error Decoding"
        
        elif 'text/html' in content_type or 'text/plain' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            title_tag = soup.find('title')
            title_text = title_tag.get_text(strip=True).lower() if title_tag else ""
            
            if target_word.lower() in title_text:
                word_found = "Yes"
            
            body = soup.find('body')
            if body:
                for element in body(["script", "style", "noscript"]):
                    element.extract()
                
                page_text = body.get_text(separator=' ', strip=True).lower()
                
                if target_word.lower() in page_text:
                    word_found = "Yes"
            
            logo_found = "N/A (Webpage)" 
            
        else:
            word_found = f"Unsupported: {content_type}"
            logo_found = "N/A"
            
    except Exception:
        word_found = "Error Connecting"
        logo_found = "Error Connecting"
        
    return [url, word_found, logo_found]

# --- Streamlit Frontend UI ---
st.title("High-Speed Bulk URL Scanner")
st.markdown("Paste image or website URLs. The script uses multithreading to scan up to 10 links simultaneously.")

col1, col2 = st.columns([1, 2])

with col1:
    input_word = st.text_input("Target Word to Scan For", value="lyve")
    input_urls = st.text_area("Paste URLs (One per line)", height=250, placeholder="https://example.com/1.jpg\nhttps://example.com/page-1")
    input_logo = st.file_uploader("Upload Reference Logo (Image URLs only)", type=["png", "jpg", "jpeg"])
    start_scan = st.button("Start Bulk Scan", type="primary", use_container_width=True)

with col2:
    if start_scan:
        if not input_urls or not input_urls.strip():
            st.warning("Please provide at least one URL.")
        elif not input_word or not input_word.strip():
            st.warning("Please provide a target word.")
        else:
            # Clean and deduplicate URLs
            target_urls = list(set([url.strip() for url in input_urls.split('\n') if url.strip()]))
            
            # Process uploaded logo into OpenCV format natively in RAM
            logo_gray = None
            if input_logo is not None:
                file_bytes = np.asarray(bytearray(input_logo.read()), dtype=np.uint8)
                logo_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            results = []
            
            # Setup visual progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            completed = 0
            total = len(target_urls)
            
            # Execute multithreaded scanning
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(process_single_url, url, logo_gray, input_word): url for url in target_urls}
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
                    completed += 1
                    # Update frontend in real time
                    progress_bar.progress(completed / total)
                    status_text.text(f"Scanned {completed} of {total} URLs...")
            
            status_text.success("Scan Complete!")
            
            # Output Results
            df = pd.DataFrame(results, columns=["URL", f"Word '{input_word}' Found", "Logo Found"])
            st.dataframe(df, use_container_width=True)
