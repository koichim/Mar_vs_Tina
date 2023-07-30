import os

src_dir = os.path.abspath("tmp/data/original")
def randomize_data(linked_dir):
    if not os.path.isdir(linked_dir):
        print(f"{linked_dir} is not a directory")

