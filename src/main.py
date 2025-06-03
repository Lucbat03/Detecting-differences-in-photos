import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Zwiększenie kontrastu
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    return img, blurred

def detect_general_differences(gray1, gray2):
    # Dynamiczne obliczanie progu na podstawie różnicy
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptacyjne progowanie
    threshold = np.mean(diff) + 2 * np.std(diff)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Zaawansowana morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    return thresholded

def detect_cracks(gray):
    # Detekcja krawędzi z adaptacyjnymi progami
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

    # Wzmocnienie linii
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Usunięcie małych obszarów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:  # Minimalny obszar
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    return mask

def detect_discolorations(img1, img2):
    # Konwersja do przestrzeni LAB dla lepszego wykrywania kolorów
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

    # Obliczanie różnicy dla kanałów A i B
    a_diff = cv2.absdiff(lab1[:,:,1], lab2[:,:,1])
    b_diff = cv2.absdiff(lab1[:,:,2], lab2[:,:,2])

    # Połączenie różnic
    color_diff = cv2.addWeighted(a_diff, 0.5, b_diff, 0.5, 0)

    # Adaptacyjne progowanie
    threshold = np.mean(color_diff) + 2 * np.std(color_diff)
    _, mask = cv2.threshold(color_diff, threshold, 255, cv2.THRESH_BINARY)

    # Morfologia
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def analyze_toothbrush_sample(dataset_path, sample_index=0):
    # Ścieżki do plików
    train_good = sorted([os.path.join(dataset_path, "train/good", f)
                         for f in os.listdir(os.path.join(dataset_path, "train/good")) if f.endswith('.png')])
    test_defect = sorted([os.path.join(dataset_path, "test/defective", f)
                          for f in os.listdir(os.path.join(dataset_path, "test/defective")) if f.endswith('.png')])
    gt_masks = sorted([os.path.join(dataset_path, "ground_truth/defective", f)
                       for f in os.listdir(os.path.join(dataset_path, "ground_truth/defective")) if f.endswith('.png')])

    # Wybór próbek
    ref_img_path = train_good[sample_index % len(train_good)]
    defect_img_path = test_defect[sample_index % len(test_defect)]
    gt_mask_path = gt_masks[sample_index % len(gt_masks)]

    # Wczytanie obrazów
    ref_img, ref_gray = load_and_preprocess(ref_img_path)
    defect_img, defect_gray = load_and_preprocess(defect_img_path)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    # Dopasowanie rozmiaru maski
    if gt_mask.shape != defect_img.shape[:2]:
        gt_mask = cv2.resize(gt_mask, (defect_img.shape[1], defect_img.shape[0]))

    # Wykrywanie zmian
    general_diff = detect_general_differences(ref_gray, defect_gray)
    cracks_ref = detect_cracks(ref_gray)
    cracks_defect = detect_cracks(defect_gray)
    new_cracks = cv2.bitwise_and(cracks_defect, cv2.bitwise_not(cracks_ref))
    discolorations = detect_discolorations(ref_img, defect_img)

    # Łączenie wyników
    combined_mask = cv2.bitwise_or(general_diff, cv2.bitwise_or(new_cracks, discolorations))

    # Filtracja małych obszarów
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(combined_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:  # Minimalny obszar
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    # Obliczanie metryk
    precision, recall = calculate_metrics(filtered_mask, gt_mask)

    # Wizualizacja wyników
    result = defect_img.copy()

    # Rysowanie wykrytych defektów
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,0,255), 2)

    # Rysowanie ground truth
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, gt_contours, -1, (0,255,0), 2)

    # Legenda
    cv2.putText(result, f"Precision: {precision:.2f}, Recall: {recall:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(result, "Detected (red) | GT (green)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    return {
        'result': result,
        'reference': ref_img,
        'defect': defect_img,
        'gt_mask': gt_mask,
        'detected_mask': filtered_mask,
        'metrics': (precision, recall)
    }

def calculate_metrics(pred_mask, gt_mask):
    # Konwersja do formatu binarnego
    gt_mask = (gt_mask > 0).astype(np.uint8).flatten()
    pred_mask = (pred_mask > 0).astype(np.uint8).flatten()

    if np.sum(gt_mask) == 0:
        return 1.0 if np.sum(pred_mask) == 0 else 0.0, 1.0

    precision = precision_score(gt_mask, pred_mask, zero_division=0)
    recall = recall_score(gt_mask, pred_mask)

    return precision, recall

def visualize_results(results):
    # Create a new figure with a specific size
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Display reference image
    axes[0, 0].imshow(cv2.cvtColor(results['reference'], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Reference Image')
    axes[0, 0].axis('off')

    # Display defective image
    axes[0, 1].imshow(cv2.cvtColor(results['defect'], cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Defective Image')
    axes[0, 1].axis('off')

    # Display results with detection
    axes[1, 0].imshow(cv2.cvtColor(results['result'], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Detection Results (Red) vs GT (Green)')
    axes[1, 0].axis('off')

    # Display detected mask
    axes[1, 1].imshow(results['detected_mask'], cmap='gray')
    axes[1, 1].set_title('Detected Defects Mask')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the figure to a temporary file and display it
    temp_file = "temp_result.png"
    plt.savefig(temp_file)
    plt.close()

    # Display the saved image using OpenCV
    img = cv2.imread(temp_file)
    if img is not None:
        cv2.imshow("Analysis Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load the temporary result image")

    print(f"Metrics - Precision: {results['metrics'][0]:.2f}, Recall: {results['metrics'][1]:.2f}")

# Przykładowe użycie
dataset_path = "photos\\toothbrush"
results = analyze_toothbrush_sample(dataset_path, sample_index=0)
visualize_results(results)