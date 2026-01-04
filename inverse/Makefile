# Clock Inverse Analyzer

.PHONY: test test-unit test-video clean exe

# Run all tests
test: test-unit test-video

# Run unit tests (fast, no video processing)
test-unit:
	python3 test_inverse.py

# Run video integration tests (requires test videos)
test-video:
	./run_all_tests.sh

# Build standalone Windows executable (requires PyInstaller)
exe:
	@command -v pyinstaller >/dev/null 2>&1 || { echo "PyInstaller not found. Install with: pip install pyinstaller"; exit 1; }
	pyinstaller --onefile --name clock_inverse \
		--hidden-import=cv2 \
		--hidden-import=numpy \
		main.py
	@echo "Executable created: dist/clock_inverse"

clean:
	rm -rf __pycache__ *.pyc build dist *.spec
