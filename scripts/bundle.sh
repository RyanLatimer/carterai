#!/bin/bash
set -euo pipefail
APP="dist/Carter.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/carterai "$APP/Contents/MacOS/Carter"
cp -R assets/licenses "$APP/Contents/Resources/Licenses"
cp THIRD_PARTY.md "$APP/Contents/Resources/Third-party notices.md"
cp assets/Info.plist "$APP/Contents/Info.plist"
cp target/mlx-metallib/mlx.metallib "$APP/Contents/Resources/mlx.metallib"
if [[ -f dist/Carter.icns ]]; then cp dist/Carter.icns "$APP/Contents/Resources/Carter.icns"; fi
if [[ -n "${SIGNING_IDENTITY:-}" ]]; then
  codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" "$APP/Contents/MacOS/Carter"
  codesign --force --options runtime --timestamp --sign "$SIGNING_IDENTITY" "$APP"
elif [[ "${DEVELOPMENT_BUILD:-0}" = 1 ]]; then
  codesign --force --sign - "$APP"
else
  echo 'Developer ID signing credentials are required. For local testing only, set DEVELOPMENT_BUILD=1.' >&2
  exit 1
fi
codesign --verify --deep --strict "$APP"
file "$APP/Contents/MacOS/Carter" | tee dist/architecture.txt
grep -q arm64 dist/architecture.txt
ditto -c -k --sequesterRsrc --keepParent "$APP" dist/Carter-Apple-Silicon.zip
