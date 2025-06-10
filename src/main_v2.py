import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

def load_and_preprocess(image_path):
    """Wczytanie i preprocessowanie obrazu"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Delikatne zwiększenie kontrastu
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Silniejsze rozmycie dla redukcji szumu
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    return img, blurred

def add_grid_overlay(image, grid_size=(10, 10), color=(0, 255, 0), thickness=2):
    """Dodanie siatki do obrazu w celu analizy regionów"""
    h, w, _ = image.shape
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]

    # Rysowanie pionowych linii siatki
    for i in range(1, grid_size[1]):
        x = i * cell_width
        cv2.line(image, (x, 0), (x, h), color, thickness)

    # Rysowanie poziomych linii siatki
    for i in range(1, grid_size[0]):
        y = i * cell_height
        cv2.line(image, (0, y), (w, y), color, thickness)

    return image

def apply_edge_detection(image):
    """Zastosowanie detekcji krawędzi (Sobel/Canny)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Użyj Sobela do detekcji krawędzi
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Wykrywanie krawędzi za pomocą Canny'ego
    canny_edges = cv2.Canny(gray, 100, 200)
    
    # Łączenie wyników Sobela i Canny'ego
    combined_edges = cv2.bitwise_or(sobel_x, sobel_y)
    
    return combined_edges, canny_edges

def detect_defects_advanced(ref_gray, defect_gray, ref_img, defect_img):
    """Zaawansowana detekcja defektów z wieloma metodami"""
    
    # 1. Maska obszaru szczoteczki
    toothbrush_mask = create_toothbrush_mask(defect_gray)
    
    # 2. Detekcja różnic intensywności z wysokim progiem
    diff = cv2.absdiff(ref_gray, defect_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # ZNACZNIE wyższy próg - tylko wyraźne różnice
    mean_diff = np.mean(diff[toothbrush_mask > 0])
    std_diff = np.std(diff[toothbrush_mask > 0])
    threshold = mean_diff + 4 * std_diff  # Zwiększone z 2 do 4
    
    _, intensity_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 3. Detekcja różnic kolorystycznych w przestrzeni LAB
    lab_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    lab_defect = cv2.cvtColor(defect_img, cv2.COLOR_BGR2LAB)
    
    # Różnice w kanałach a* i b* (kolory)
    a_diff = cv2.absdiff(lab_ref[:,:,1], lab_defect[:,:,1])
    b_diff = cv2.absdiff(lab_ref[:,:,2], lab_defect[:,:,2])
    
    # Kombinacja różnic kolorystycznych
    color_diff = np.maximum(a_diff, b_diff)
    color_diff = cv2.GaussianBlur(color_diff, (5, 5), 0)
    
    # Wysoki próg dla kolorów
    mean_color = np.mean(color_diff[toothbrush_mask > 0])
    std_color = np.std(color_diff[toothbrush_mask > 0])
    color_threshold = mean_color + 3 * std_color
    
    _, color_mask = cv2.threshold(color_diff, color_threshold, 255, cv2.THRESH_BINARY)
    
    # 4. Detekcja lokalnych anomalii (blob detection)
    # Wykorzystanie różnicy po rozmyciu do znajdowania lokalnych zmian
    ref_blur_heavy = cv2.GaussianBlur(ref_gray, (31, 31), 0)
    defect_blur_heavy = cv2.GaussianBlur(defect_gray, (31, 31), 0)
    
    local_diff = cv2.absdiff(ref_blur_heavy, defect_blur_heavy)
    mean_local = np.mean(local_diff[toothbrush_mask > 0])
    std_local = np.std(local_diff[toothbrush_mask > 0])
    local_threshold = mean_local + 3 * std_local
    
    _, local_mask = cv2.threshold(local_diff, local_threshold, 255, cv2.THRESH_BINARY)
    
    # 5. Kombinacja wszystkich masek
    combined_mask = cv2.bitwise_or(intensity_mask, color_mask)
    combined_mask = cv2.bitwise_or(combined_mask, local_mask)
    
    # Zastosowanie maski szczoteczki
    combined_mask = cv2.bitwise_and(combined_mask, toothbrush_mask)
    
    # 6. Agresywne czyszczenie - tylko duże, znaczące obszary
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    
    # Zamknięcie małych dziur
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean)
    
    # Usunięcie małych obiektów
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # 7. Filtracja konturów po wielkości
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(combined_mask)
    
    min_area = 500  # ZNACZNIE zwiększony minimalny obszar
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Dodatkowo sprawdź aspect ratio i wypełnienie
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if 0.2 < aspect_ratio < 5.0:
                cv2.fillPoly(filtered_mask, [cnt], 255)
    
    return filtered_mask

def calculate_metrics(pred_mask, gt_mask):
    """Obliczanie precision i recall"""
    gt_binary = (gt_mask > 127).astype(np.uint8)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    
    gt_flat = gt_binary.flatten()
    pred_flat = pred_binary.flatten()
    
    if np.sum(gt_flat) == 0 and np.sum(pred_flat) == 0:
        return 1.0, 1.0  # Perfect match when both are empty
    elif np.sum(gt_flat) == 0:
        return 0.0, 1.0  # No true positives, but recall is undefined/perfect
    
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    
    return precision, recall

def create_toothbrush_mask(gray_img):
    """Utworzenie maski obszaru szczoteczki do zębów"""
    # Otsu thresholding dla separacji szczoteczki od tła
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Znalezienie największego konturu (szczoteczka)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.ones_like(gray_img) * 255
    
    # Największy kontur
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Utworzenie maski
    mask = np.zeros_like(gray_img)
    cv2.fillPoly(mask, [largest_contour], 255)
    
    # Erozja maski aby uniknąć krawędzi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.erode(mask, kernel, iterations=1)
    
    return mask

def visualize_results(results):
    """Wizualizacja wyników analizy"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(cv2.cvtColor(results['reference'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reference Image (Good)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(results['defect'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Defective Image', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(results['gt_mask'], cmap='gray')
    axes[0, 2].set_title('Ground Truth Mask', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(results['detected_mask'], cmap='gray')
    axes[1, 0].set_title('Detected Defects Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    comparison = np.zeros((*results['gt_mask'].shape, 3))
    comparison[:,:,1] = (results['gt_mask'] > 127).astype(float)  # GT w zieleni
    comparison[:,:,0] = (results['detected_mask'] > 127).astype(float)  # Detekcja w czerwieni
    axes[1, 1].imshow(comparison)
    axes[1, 1].set_title('Mask Comparison\n(Green=GT, Red=Detected, Yellow=Overlap)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Final Result\nPrecision: {results["metrics"][0]:.3f}, Recall: {results["metrics"][1]:.3f}', 
                        fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal Metrics:")
    print(f"Precision: {results['metrics'][0]:.3f}")
    print(f"Recall: {results['metrics'][1]:.3f}")
    print(f"F1-Score: {2 * (results['metrics'][0] * results['metrics'][1]) / (results['metrics'][0] + results['metrics'][1]):.3f}")

def analyze_toothbrush_sample(dataset_path, sample_index=0):
    """Analiza próbki szczoteczki z poprawioną detekcją"""
    
    train_good = sorted([os.path.join(dataset_path, "train/good", f)
                         for f in os.listdir(os.path.join(dataset_path, "train/good")) if f.endswith('.png')])
    test_defect = sorted([os.path.join(dataset_path, "test/defective", f)
                          for f in os.listdir(os.path.join(dataset_path, "test/defective")) if f.endswith('.png')])
    gt_masks = sorted([os.path.join(dataset_path, "ground_truth/defective", f)
                         for f in os.listdir(os.path.join(dataset_path, "ground_truth/defective")) if f.endswith('.png')])
    
    ref_img_path = train_good[sample_index % len(train_good)]
    defect_img_path = test_defect[sample_index % len(test_defect)]
    gt_mask_path = gt_masks[sample_index % len(gt_masks)]
    
    ref_img, ref_gray = load_and_preprocess(ref_img_path)
    defect_img, defect_gray = load_and_preprocess(defect_img_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if gt_mask.shape != defect_img.shape[:2]:
        gt_mask = cv2.resize(gt_mask, (defect_img.shape[1], defect_img.shape[0]))
    
    detected_mask = detect_defects_advanced(ref_gray, defect_gray, ref_img, defect_img)
    precision, recall = calculate_metrics(detected_mask, gt_mask)
    
    result = defect_img.copy()
    
    contours, _ = cv2.findContours(detected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(result, 'D', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, gt_contours, -1, (0, 255, 0), 3)
    
    cv2.putText(result, f"Precision: {precision:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(result, f"Recall: {recall:.3f}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return {
        'result': result,
        'reference': ref_img,
        'defect': defect_img,
        'gt_mask': gt_mask,
        'detected_mask': detected_mask,
        'metrics': (precision, recall)
    }

def batch_analysis(dataset_path, num_samples=5):
    """Analiza wielu próbek do oceny ogólnej wydajności z wizualizacją."""
    all_precisions = []
    all_recalls = []
    
    print("Starting batch analysis...")
    
    for i in range(num_samples):
        try:
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            results = analyze_toothbrush_sample(dataset_path, sample_index=i)
            precision, recall = results['metrics']
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            # Wizualizacja wyników dla każdej próbki
            visualize_results(results)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if all_precisions:
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"\n=== BATCH RESULTS ===")
        print(f"Average Precision: {avg_precision:.3f} (±{np.std(all_precisions):.3f})")
        print(f"Average Recall: {avg_recall:.3f} (±{np.std(all_recalls):.3f})")
        print(f"Average F1-Score: {avg_f1:.3f}")
        print(f"Samples processed: {len(all_precisions)}")
    
    return all_precisions, all_recalls

# Przykład użycia
if __name__ == "__main__":
    dataset_path = "photos/toothbrush"
    
    # Analiza pojedynczej próbki
    print("=== SINGLE SAMPLE ANALYSIS ===")
    results = analyze_toothbrush_sample(dataset_path, sample_index=0)
    visualize_results(results)
    
    # Analiza wsadowa (opcjonalna)
    print("\n=== BATCH ANALYSIS ===")
    batch_analysis(dataset_path, num_samples=3)

