import os
import glob

def generate(input_dir):
    file_start = "# __init__.py\n__all__ = [\n"
    file_end = "]"
    dir = os.path.join(input_dir, "*.py")
    pths = glob.glob(dir)
    names = []
    file = file_start
    for pth in pths:
        [_, name] = os.path.split(pth)
        [prefix, _] = os.path.splitext(name)
        if prefix != "__init__":
            names.append(prefix)
            file += "\t\"{}\",\n".format(prefix)
    file += file_end
    print(file)
    return [names, file]
    
def make_all(dir_mother):
    dirs = glob.glob(os.path.join(dir_mother, "*/"))
    for dir in dirs:
        [names, file] = generate(dir)
        if names != "":
            with open(os.path.join(dir, "__init__.py"), "w") as f:
                f.write(file)

if __name__ == "__main__":  
    make_all("C:/Users/Li/Desktop/Pose6dSolver-pyqt/src/")