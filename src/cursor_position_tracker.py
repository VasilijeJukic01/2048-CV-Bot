import pyautogui
import keyboard


def print_cursor_position():
    x, y = pyautogui.position()
    print(f"Cursor position: ({x}, {y})")


print("Press 's' to print cursor position. Press 'q' to quit.")
while True:
    if keyboard.is_pressed('s'):
        print_cursor_position()
        while keyboard.is_pressed('s'):
            pass
    elif keyboard.is_pressed('q'):
        break
