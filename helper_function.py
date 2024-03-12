import cv2


class Helper:
    def __init__(self) -> None:
        self.data = {}

    def text(self, image, x, y, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,
                    f'{text}',
                    (x, y),
                    font, 1,
                    (50, 50, 225),
                    2,
                    cv2.LINE_4)
