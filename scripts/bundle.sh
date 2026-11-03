#!/bin/bash
set -euo pipefail
APP="dist/Carter.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/carterai "$APP/Contents/MacOS/Carter"
cp -R assets/licenses "$APP/Contents/Resources/Licenses"
cp THIRD_PARTY.md "$APP/Contents/Resources/Third-party notices.md"
cp assets/Info.plist "$APP/Contents/Info.plist"
if [[ -f dist/Carter.icns ]]; then cp dist/Carter.icns "$APP/Contents/Resources/Carter.icns"; fi
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
file "$APP/Contents/MacOS/Carter" | tee dist/architecture.txt
grep -q arm64 dist/architecture.txt
# Preserve the app's executable permissions and macOS bundle metadata.
ditto -c -k --sequesterRsrc --keepParent "$APP" dist/Carter-Apple-Silicon.zip
