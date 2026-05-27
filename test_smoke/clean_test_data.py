'''Wipe transient smoke outputs under test_data/. The smoke tests here are
purely in-memory (no artifacts written), so the list is empty for now; keep
the file as the family-standard entry point for future visual artifacts.'''

import os
import shutil

LIST_PATH_CLEAN: list[str] = []


def main() -> None:
    for path in LIST_PATH_CLEAN:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print('removed dir: %s' % path)
        elif os.path.isfile(path):
            os.remove(path)
            print('removed file: %s' % path)


if __name__ == '__main__':
    main()
