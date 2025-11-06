from pathlib import Path
import subprocess
import sys

RAW_DIR = Path("data/raw/fakenewsnet_min")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Downloading ISOT Fake News Dataset from Google Drive")
print("=" * 60)
print()

# Install gdown if not available
try:
    import gdown
    print("✓ gdown library found")
except ImportError:
    print("Installing gdown library...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--break-system-packages"])
        import gdown
        print("✓ gdown installed successfully")
    except:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
        print("✓ gdown installed successfully")

print()

# Google Drive sharing links
# REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE SHARING LINKS
FAKE_URL = "https://drive.google.com/file/d/1CeQQmQrafEWQgnSSX-rFoz_67GA0jWaX/view?usp=drive_link"
TRUE_URL = "https://drive.google.com/file/d/13dAQe4y20WGM0Jc9ypsZ87RopNdlspPb/view?usp=drive_link"

# Check if URLs are configured
if "YOUR_GOOGLE_DRIVE_LINK" in FAKE_URL or "YOUR_GOOGLE_DRIVE_LINK" in TRUE_URL:
    print("✗ ERROR: Google Drive URLs not configured!")
    print("\n" + "=" * 60)
    print("SETUP INSTRUCTIONS:")
    print("=" * 60)
    print("\n1. Upload Fake.csv and True.csv to Google Drive")
    print("\n2. For each file:")
    print("   - Right-click → Share")
    print("   - Change to 'Anyone with the link'")
    print("   - Copy the sharing link")
    print("\n3. Edit scripts/download_fnn.py:")
    print("   Replace the FAKE_URL and TRUE_URL with your links")
    print("\n4. Example:")
    print('   FAKE_URL = "https://drive.google.com/file/d/1ABC123.../view?usp=sharing"')
    print('   TRUE_URL = "https://drive.google.com/file/d/1XYZ789.../view?usp=sharing"')
    print("\n" + "=" * 60)
    sys.exit(1)

# Download Fake.csv
print("Downloading Fake.csv from Google Drive...")
fake_path = RAW_DIR / "Fake.csv"
try:
    gdown.download(FAKE_URL, str(fake_path), quiet=False, fuzzy=True)
    if fake_path.exists():
        fake_size = fake_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded Fake.csv ({fake_size:.2f} MB)")
    else:
        raise Exception("File not found after download")
except Exception as e:
    print(f"✗ Failed to download Fake.csv: {e}")
    print("\nTroubleshooting:")
    print("  - Make sure the file is shared as 'Anyone with the link'")
    print("  - Check that the URL is correct")
    print("  - Try downloading manually and placing in:", RAW_DIR.absolute())
    sys.exit(1)

print()

# Download True.csv
print("Downloading True.csv from Google Drive...")
true_path = RAW_DIR / "True.csv"
try:
    gdown.download(TRUE_URL, str(true_path), quiet=False, fuzzy=True)
    if true_path.exists():
        true_size = true_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded True.csv ({true_size:.2f} MB)")
    else:
        raise Exception("File not found after download")
except Exception as e:
    print(f"✗ Failed to download True.csv: {e}")
    print("\nTroubleshooting:")
    print("  - Make sure the file is shared as 'Anyone with the link'")
    print("  - Check that the URL is correct")
    print("  - Try downloading manually and placing in:", RAW_DIR.absolute())
    sys.exit(1)

print()
print("=" * 60)
print("✓ SUCCESS: Dataset downloaded!")
print("=" * 60)
print(f"\nFiles saved in: {RAW_DIR.absolute()}")
print(f"  - Fake.csv: {fake_size:.2f} MB")
print(f"  - True.csv: {true_size:.2f} MB")
print("\nReady to run: python scripts/prepare_fnn.py")