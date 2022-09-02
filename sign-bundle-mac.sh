#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <developer_email> <hash_developer_id_application>"
    exit 1;
fi

DEVELOPER_EMAIL="$1"
HASH_DEVELOPER_ID_APPLICATION="$2"

# To find hash of your cert run:
# security find-identity -p basic -v

# To add your App-Specific key run:
# xcrun altool --store-password-in-keychain-item "AC_PASSWORD" -u "your-username" -p "your-password"

codesign --deep --force --options=runtime --entitlements ./entitlements.plist --sign "$HASH_DEVELOPER_ID_APPLICATION" --timestamp ./dist/godr.app
ditto -c -k --keepParent "dist/godr.app" dist/godr-notarize-stage.zip
xcrun altool --notarize-app -t osx -f dist/godr-notarize-stage.zip --primary-bundle-id gle8098.godr -u "$DEVELOPER_EMAIL" --password "@keychain:AC_PASSWORD"

echo "Now wait until Apple sends you an email with notarizing results..."
read -p "Press enter to continue"

rm dist/godr-notarize-stage.zip
xcrun stapler staple "dist/godr.app"
spctl --assess --type execute -vvv "dist/godr.app"
