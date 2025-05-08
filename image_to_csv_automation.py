import cv2
import numpy as np
import pandas as pd
import pytesseract
from matplotlib import pyplot as plt

def grid_detection(image_path, output_csv="output.csv", debug=True):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image at {image_path}")
        return

    visual_image = original_image.copy()

    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    if debug:
        cv2.imwrite("binary.jpg", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    if debug:
        cv2.imwrite("contours.jpg", contour_image)

    cell_contours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if 2000 < area < 50000:
                cell_contours.append((x, y, w, h))            

    cell_image = original_image.copy()
    for x, y, w, h in cell_contours:
        cv2.rectangle(cell_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    if debug:
        cv2.imwrite("detected_cells.jpg", cell_image)

    if not cell_contours:
        print("No cells detected using contour method, trying grid line detection...")

        edges = cv2.Canny(binary, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 50,
            minLineLength=30,
            maxLineGap=20
        )

        line_image = original_image.copy()

        h_lines = []
        v_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if abs(y2 - y1) < abs(x2 - x1):
                    h_lines.append((min(x1, x2), max(x1, x2), (y1 + y2) // 2))
                else:
                    v_lines.append(((x1 + x2) // 2, min(y1, y2), max(y1, y2)))

        if debug:
            cv2.imwrite("lines.jpg", line_image)

        h_lines.sort(key=lambda x: x[2])
        v_lines.sort(key=lambda x: x[0])

        def cluster_lines(lines, axis_idx, threshold=30):
            if not lines:
                return []

            clusters = [[lines[0]]]

            for line in lines[1:]:
                if line[axis_idx] - clusters[-1][-1][axis_idx] > threshold:
                    clusters.append([line])
                else:
                    clusters[-1].append(line)

            result = []
            for cluster in clusters:
                if axis_idx == 2:
                    avg = sum(line[2] for line in cluster) // len(cluster)
                    x_min = min(line[0] for line in cluster)
                    x_max = max(line[1] for line in cluster)
                    result.append((x_min, x_max, avg))
                else:
                    avg = sum(line[0] for line in cluster) // len(cluster)
                    y_min = min(line[1] for line in cluster)
                    y_max = max(line[2] for line in cluster)
                    result.append((avg, y_min, y_max))

            return result

        h_lines = cluster_lines(h_lines, 2)
        v_lines = cluster_lines(v_lines, 0)

        clustered_line_image = original_image.copy()
        for x1, x2, y in h_lines:
            cv2.line(clustered_line_image, (x1, y), (x2, y), (0, 255, 0), 2)
        for x, y1, y2 in v_lines:
            cv2.line(clustered_line_image, (x, y1), (x, y2), (255, 0, 0), 2)

        if debug:
            cv2.imwrite("clustered_lines.jpg", clustered_line_image)

        cells = []
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                x1 = v_lines[j][0]
                y1 = h_lines[i][2]
                x2 = v_lines[j+1][0]
                y2 = h_lines[i+1][2]

                cell = (x1, y1, x2 - x1, y2 - y1)
                cells.append(cell)
                cv2.rectangle(clustered_line_image, (x1, y1), (x1 + (x2-x1), y1 + (y2-y1)), (0, 255, 255), 2)

        if debug:
            cv2.imwrite("cell_grid.jpg", clustered_line_image)

        cell_contours = cells

    if not cell_contours:
        print("No grid cells detected.")
        return

    def cluster_coordinates(coords, tolerance=20):
        if not coords:
            return []

        coords = sorted(coords)
        clusters = [[coords[0]]]

        for coord in coords[1:]:
            if coord - clusters[-1][-1] > tolerance:
                clusters.append([coord])
            else:
                clusters[-1].append(coord)

        return [sum(cluster) // len(cluster) for cluster in clusters]

    y_coords = [cell[1] for cell in cell_contours]
    x_coords = [cell[0] for cell in cell_contours]

    y_clusters = cluster_coordinates(y_coords)
    x_clusters = cluster_coordinates(x_coords)

    grid = {}
    for cell in cell_contours:
        x, y, w, h = cell

        y_diffs = [abs(y - cluster) for cluster in y_clusters]
        row_idx = y_diffs.index(min(y_diffs))

        x_diffs = [abs(x - cluster) for cluster in x_clusters]
        col_idx = x_diffs.index(min(x_diffs))

        grid[(row_idx, col_idx)] = (x, y, w, h)

    max_row = max([key[0] for key in grid.keys()]) if grid else -1
    max_col = max([key[1] for key in grid.keys()]) if grid else -1

    grid_image = original_image.copy()

    table = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    for (row_idx, col_idx), (x, y, w, h) in grid.items():
        cv2.rectangle(grid_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(grid_image, f"{row_idx},{col_idx}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        padding = 10
        cell_img = gray[y+padding:y+h-padding, x+padding:x+w-padding]

        if cell_img.size == 0 or cell_img.shape[0] < 10 or cell_img.shape[1] < 10:
            continue

        _, cell_thresh = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if debug:
            cv2.imwrite(f"cell_{row_idx}_{col_idx}.jpg", cell_thresh)

        try:
            text = pytesseract.image_to_string(
                cell_thresh,
                config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ).strip()

            if text or text.isdigit():
                table[row_idx][col_idx] = text
                cv2.putText(grid_image, text, (x+w//2-10, y+h//2+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        except Exception as e:
            print(f"OCR error for cell {row_idx},{col_idx}: {e}")

    if debug:
        cv2.imwrite("final_grid.jpg", grid_image)

    df = pd.DataFrame(table)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Saved table to {output_csv}")

    print(f"Detected grid shape: {max_row+1} rows × {max_col+1} columns")
    print(df.to_string(index=False, header=False))
    return df



# the program is under development.. so it may show inaccurate result
if __name__ == "__main__":
    grid_detection("test.png", "output_improved.csv")
