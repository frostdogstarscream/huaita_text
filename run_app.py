import sys


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        from package_self_test import run_self_test

        raise SystemExit(run_self_test())

    from main import run_server

    run_server()
