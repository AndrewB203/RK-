import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = r"C:\Users\bolsh\OneDrive\Gear.jpg"  # 
img = cv2.imread(img_path)

if img is None:
    print(" Изображение не найдено! Проверьте путь.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)



# ========================
# 3. Поиск окружностей (Hough Circle Transform)
# ========================


circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=225,
    param1=1019,
    param2=50,      #  можно уменьшить до 25–28, если не находятся колёса
    minRadius=60,   #  радиус колеса ~60-180 пикселей
    maxRadius=120
)


# ========================
# 4. Рисование окружностей и центров
# ========================

img_with_circles = img_rgb.copy()  # копия для рисования

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Окружность — зелёная
        cv2.circle(img_with_circles, (x, y), r, (0, 255, 0), 3)
        # Центр — красная точка
        cv2.circle(img_with_circles, (x, y), 2, (0, 0, 255), 3)
else:
    print(" Окружности не найдены. Попробуйте изменить параметры.")


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title('Оригинал')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Границы (Canny)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_with_circles)
plt.title('Найденные колёса (зелёные окружности)')
plt.axis('off')

plt.tight_layout()
plt.show()



cv2.imwrite("detected_wheels.jpg", cv2.cvtColor(img_with_circles, cv2.COLOR_RGB2BGR))
print("Результат сохранён как 'detected_wheels.jpg'")