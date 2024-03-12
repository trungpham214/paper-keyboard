import cv2
import string
import pytesseract
import helper_function


helper = helper_function.Helper()

img = cv2.imread('keyboard1.png')
keyboard_char = list(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hImg, wImg, _ = img.shape

boxes = pytesseract.image_to_boxes(img)

array1 = []
char_boxes = []
wordslist = ['ESC', 'Tab', 'Capslock','Shift', 'Crtl', 'Fn', 'Alt', 'Space', 'Enter',
             'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12']

virtual_keyboard = []

boxes = boxes.splitlines()

for b in boxes:
    b = b.split(' ')
    if b[0] in keyboard_char:
        array1.append(b[0])
        char_boxes.append(b)

for word in wordslist:
    # if word == 'Fn':
        array2 = [char for char in word]
        # print(array2)
        p1 = 0
        p2 = 0

        for p1 in range(0, len(array1)):
            if array1[p1] == array2[p2]:
                p2 += 1
            else:
                p2 = 0

            if p2 == len(word):
                start = p1 - p2 + 1
                end = p1 + 1
                # print(array1[p1 - p2 + 1: p1 + 1])
                # print(array1[start])
                b = char_boxes[start]
                (x, y, w, h) = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                # cv2.rectangle(img, (x, hImg - y), (x + 100, hImg - y - 100), (0, 0, 255), 3)
                virtual_keyboard.append([word, x, y, w, h])
                for i in range(start, end):
                    char_boxes[i][0] = word
                break

for b in char_boxes:
    if b[0] not in wordslist:
        (x, y, w, h) = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        virtual_keyboard.append([b[0], x, y, w, h])

for i in virtual_keyboard:
    if i[0] =='s':
        virtual_keyboard.remove(i)

print(virtual_keyboard)

class Keyboard:
    def __init__(self) -> None:
        self.data = {}

    def drawKeyboard(self, image):
        for coor in virtual_keyboard:
            (x, y, w, h) = coor[1], coor[2], coor[3], coor[4]
            cv2.rectangle(image, (x, hImg - y), (x + 100, hImg - y - 100), (0, 0, 255), 3)
            helper.text(image, x + 5, hImg - y - 5, coor[0])

    def getKeyboard(self):
        return virtual_keyboard
