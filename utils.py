import os

def clear_console():
    os.system(f'cls' if os.name == 'nt' else 'clear')