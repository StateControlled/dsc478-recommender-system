import sys
import argparse


def main(argv: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(description="description")

    print(f'Hi')

    return 0


if __name__ == '__main__':
    sys.exit(main())
